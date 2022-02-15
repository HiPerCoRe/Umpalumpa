#include <libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>

namespace umpalumpa::fourier_processing {

namespace {// to avoid poluting
  inline static const auto kKernelFile =
    utils::GetSourceFilePath("libumpalumpa/algorithms/fourier_processing/fp_cuda_kernels.cu");

  struct Strategy1 : public FPCUDA::KTTStrategy
  {
    // Inherit constructor
    using FPCUDA::KTTStrategy::KTTStrategy;

    // FIXME improve name of the kernel and variable
    static constexpr auto kTMP = "scaleFFT2DKernel";
    // Currently we create one thread per each pixel of a single image. Each thread processes
    // same pixel of all images. The other option for 2D images is to map N dimension to the
    // Z dimension, ie. create more threads, each thread processing fewer images.

    size_t GetHash() const override { return 0; }

    std::unique_ptr<tuning::Leader> CreateLeader() const override
    {
      return tuning::StrategyGroup::CreateLeader(*this, alg);
    }
    tuning::StrategyGroup LoadTuningData() const override
    {
      return tuning::StrategyGroup::LoadTuningData(*this, alg);
    }

    std::vector<ktt::KernelConfiguration> GetDefaultConfigurations() const override
    {
      return { kttHelper.GetTuner().CreateConfiguration(GetKernelId(),
        { { "blockSizeX", static_cast<uint64_t>(32) },
          { "blockSizeY", static_cast<uint64_t>(8) } }) };
    }

    bool IsEqualTo(const TunableStrategy &ref) const override
    {
      bool isEqual = true;
      try {
        auto &refStrat = dynamic_cast<const Strategy1 &>(ref);
        isEqual = isEqual && GetOutputRef().IsEquivalentTo(refStrat.GetOutputRef());
        isEqual = isEqual && GetInputRef().IsEquivalentTo(refStrat.GetInputRef());
        isEqual = isEqual && GetSettings().IsEquivalentTo(refStrat.GetSettings());
        // Size.n has to be also equal for true equality
        isEqual = isEqual
                  && GetInputRef().GetData().info.GetSize()
                       == refStrat.GetInputRef().GetData().info.GetSize();
      } catch (std::bad_cast &) {
        isEqual = false;
      }
      return isEqual;
    }

    bool IsSimilarTo(const TunableStrategy &ref) const override
    {
      bool isSimilar = true;
      try {
        auto &refStrat = dynamic_cast<const Strategy1 &>(ref);
        isSimilar = isSimilar && GetOutputRef().IsEquivalentTo(refStrat.GetOutputRef());
        isSimilar = isSimilar && GetInputRef().IsEquivalentTo(refStrat.GetInputRef());
        isSimilar = isSimilar && GetSettings().IsEquivalentTo(refStrat.GetSettings());
        // Using naive similarity: same as equality except for ignoring Size.n
        // TODO real similarity check
      } catch (std::bad_cast &) {
        isSimilar = false;
      }
      return isSimilar;
    }

    bool InitImpl() override
    {
      // FIXME check settings
      const auto &out = alg.Get().GetOutputRef();
      const auto &in = alg.Get().GetInputRef();
      if (!AFP::IsFloat(out, in)) return false;

      const auto &size = out.GetData().info.GetPaddedSize();
      auto &tuner = kttHelper.GetTuner();

      // ensure that we have the kernel loaded to KTT
      // this has to be done in critical section, as multiple instances of this algorithm
      // might run on the same worker
      const auto &s = alg.Get().GetSettings();
      // std::lock_guard<std::mutex> lck(kttHelper.GetMutex());
      AddKernelDefinition(kTMP,
        kKernelFile,
        ktt::DimensionVector{ size.x, size.y, size.z },
        { std::to_string(s.GetApplyFilter()),
          std::to_string(s.GetNormalize()),
          std::to_string(s.GetCenter()),
          std::to_string(s.GetMaxFreq().has_value()),
          std::to_string(s.GetShift()) });
      auto definitionId = GetDefinitionId();

      AddKernel(kTMP, definitionId);
      auto kernelId = GetKernelId();

      tuner.AddParameter(
        kernelId, "blockSizeX", std::vector<uint64_t>{ 1, 2, 4, 8, 16, 32, 64, 128 });
      tuner.AddParameter(
        kernelId, "blockSizeY", std::vector<uint64_t>{ 1, 2, 4, 8, 16, 32, 64, 128 });

      tuner.AddConstraint(
        kernelId, { "blockSizeX", "blockSizeY" }, [&tuner](const std::vector<uint64_t> &params) {
          return params[0] * params[1] <= tuner.GetCurrentDeviceInfo().GetMaxWorkGroupSize();
        });

      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Local,
        ktt::ModifierDimension::X,
        "blockSizeX",
        ktt::ModifierAction::Multiply);
      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Local,
        ktt::ModifierDimension::Y,
        "blockSizeY",
        ktt::ModifierAction::Multiply);

      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Global,
        ktt::ModifierDimension::X,
        "blockSizeX",
        ktt::ModifierAction::DivideCeil);
      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Global,
        ktt::ModifierDimension::Y,
        "blockSizeY",
        ktt::ModifierAction::DivideCeil);

      tuner.SetSearcher(kernelId, std::make_unique<ktt::RandomSearcher>());
      return true;
    }

    std::string GetName() const override { return "Strategy1"; }

    bool ExecuteImpl(const FPCUDA::OutputData &out, const FPCUDA::InputData &in) override
    {
      if (!in.GetData().IsValid() || in.GetData().IsEmpty() || !out.GetData().IsValid()
          || out.GetData().IsEmpty())
        return false;

      auto &tuner = kttHelper.GetTuner();
      // prepare input data
      auto argIn = AddArgumentVector<float2>(in.GetData(), ktt::ArgumentAccessType::ReadOnly);

      // prepare output data
      auto argOut = AddArgumentVector<float2>(out.GetData(), ktt::ArgumentAccessType::WriteOnly);

      auto inSize = tuner.AddArgumentScalar(in.GetData().info.GetSize());
      auto inSpatialSize = tuner.AddArgumentScalar(in.GetData().info.GetSpatialSize());
      auto outSize = tuner.AddArgumentScalar(out.GetData().info.GetSize());

      const auto &s = alg.Get().GetSettings();
      auto filter = [&s, &tuner, &in, this]() {
        if (s.GetApplyFilter()) {
          return AddArgumentVector<float>(in.GetFilter(), ktt::ArgumentAccessType::ReadOnly);
        }
        return tuner.AddArgumentScalar(NULL);
      }();

      // normalize using the original size
      auto normFactor =
        tuner.AddArgumentScalar(static_cast<float>(in.GetData().info.GetNormFactor()));

      auto maxFreq = tuner.AddArgumentScalar(s.GetMaxFreq().value_or(0));

      auto kernelId = GetKernelId();

      SetArguments(GetDefinitionId(),
        { argIn, argOut, inSize, inSpatialSize, outSize, filter, normFactor, maxFreq });

      auto &size = out.GetData().info.GetPaddedSize();
      tuner.SetLauncher(kernelId, [this, &size](ktt::ComputeInterface &interface) {
        auto definitionId = GetDefinitionId();
        auto blockDim = interface.GetCurrentLocalSize(definitionId);
        ktt::DimensionVector gridDim(size.x, size.y, size.z);
        gridDim.RoundUp(blockDim);
        gridDim.Divide(blockDim);
        if (ShouldBeTuned(GetKernelId())) {
          interface.RunKernel(definitionId, gridDim, blockDim);
        } else {
          interface.RunKernelAsync(definitionId, interface.GetAllQueues().at(0), gridDim, blockDim);
        }
      });

      ExecuteKernel(kernelId);

      return true;
    };
  };
}// namespace

void FPCUDA::Synchronize()
{
  // FIXME when queues are implemented, synchronize only used queues
  GetHelper().GetTuner().SynchronizeDevice();
}

std::vector<std::unique_ptr<FPCUDA::Strategy>> FPCUDA::GetStrategies() const
{
  std::vector<std::unique_ptr<FPCUDA::Strategy>> vec;
  vec.emplace_back(std::make_unique<Strategy1>(*this));
  return vec;
}

}// namespace umpalumpa::fourier_processing
