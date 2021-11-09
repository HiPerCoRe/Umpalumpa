#include <libumpalumpa/algorithms/correlation/correlation_cuda.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>

namespace umpalumpa::correlation {

namespace {// to avoid poluting
  inline static const auto kKernelFile =
    utils::GetSourceFilePath("libumpalumpa/algorithms/correlation/correlation_cuda_kernels.cu");

  struct Strategy1 final : public Correlation_CUDA::KTTStrategy
  {
    // Inherit constructor
    using Correlation_CUDA::KTTStrategy::KTTStrategy;

    // FIXME improve name of the kernel and this variable
    static constexpr auto kTMP = "correlate2D";
    // Currently we create one thread per each pixel of a single image. Each thread processes
    // same pixel of all images. The other option for 2D images is to map N dimension to the
    // Z dimension, ie. create more threads, each thread processing fewer images.

    size_t GetHash() const override { return 0; }
    bool IsSimilarTo(const TunableStrategy &ref) const override
    {
      bool similar = false;
      // TODO move try-catch somewhere else
      try {
        // FIXME refactor
        auto &refAlg = dynamic_cast<const Strategy1 &>(ref).alg.Get();
        auto &thisAlg = this->alg.Get();
        auto refSize1 = refAlg.GetInputRef().GetData1().info.GetSize();
        auto thisSize1 = thisAlg.GetInputRef().GetData1().info.GetSize();
        auto refSize2 = refAlg.GetInputRef().GetData2().info.GetSize();
        auto thisSize2 = thisAlg.GetInputRef().GetData2().info.GetSize();
        // NOTE for testing size equivalence means similarity
        similar = thisSize1.IsEquivalentTo(refSize1) && thisSize2.IsEquivalentTo(refSize2);
        // TODO real similarity check
      } catch (std::bad_cast &) {
        similar = false;
      }
      return similar;
    }

    bool InitImpl() override
    {
      // FIXME check settings
      const auto &out = alg.Get().GetOutputRef();
      const auto &in = alg.Get().GetInputRef();
      bool canProcess = ACorrelation::IsFloat(out, in);
      if (!canProcess) return false;

      const auto &size = out.GetCorrelations().info.GetPaddedSize();
      auto &tuner = kttHelper.GetTuner();

      // ensure that we have the kernel loaded to KTT
      // this has to be done in critical section, as multiple instances of this algorithm
      // might run on the same worker
      const auto &s = alg.Get().GetSettings();
      std::lock_guard<std::mutex> lck(kttHelper.GetMutex());
      definitionId = GetKernelDefinitionId(kTMP,
        kKernelFile,
        ktt::DimensionVector{ size.x, size.y, size.z },
        { "float2", std::to_string(s.GetCenter()) });
      kernelId = tuner.CreateSimpleKernel(kTMP + std::to_string(strategyId), definitionId);

      tuner.AddParameter(kernelId, "TILE", std::vector<uint64_t>{ 1, 2, 4, 8 });

      tuner.AddParameter(
        kernelId, "blockSizeX", std::vector<uint64_t>{ 1, 2, 4, 8, 16, 32, 64, 128 });
      tuner.AddParameter(
        kernelId, "blockSizeY", std::vector<uint64_t>{ 1, 2, 4, 8, 16, 32, 64, 128 });

      tuner.AddConstraint(
        kernelId, { "blockSizeX", "blockSizeY" }, [&tuner](const std::vector<uint64_t> &params) {
          return params[0] * params[1] <= tuner.GetCurrentDeviceInfo().GetMaxWorkGroupSize();
        });
      tuner.AddConstraint(kernelId,
        { "blockSizeX", "TILE" },
        [](const std::vector<uint64_t> &params) { return params[0] >= params[1]; });

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
      return canProcess;
    }

    std::string GetName() const override final { return "Strategy1"; }

    bool Execute(const Correlation_CUDA::OutputData &out,
      const Correlation_CUDA::InputData &in) override final
    {
      if (!in.GetData1().IsValid() || in.GetData1().IsEmpty()
          || !out.GetCorrelations().IsValid()// FIXME refactor
          || out.GetCorrelations().IsEmpty())
        return false;

      auto &tuner = kttHelper.GetTuner();
      // prepare input GetData1()
      auto argIn1 = AddArgumentVector<float2>(in.GetData1(), ktt::ArgumentAccessType::ReadOnly);

      auto argIn2 = AddArgumentVector<float2>(in.GetData2(), ktt::ArgumentAccessType::ReadOnly);

      auto argOut =
        AddArgumentVector<float2>(out.GetCorrelations(), ktt::ArgumentAccessType::WriteOnly);

      auto inSize = tuner.AddArgumentScalar(in.GetData1().info.GetSize());
      auto in2N = tuner.AddArgumentScalar(static_cast<int>(in.GetData2().info.GetSize().n));
      // FIXME this would be better as kernel template argument
      auto isWithin =
        tuner.AddArgumentScalar(static_cast<int>(in.GetData1().GetPtr() == in.GetData2().GetPtr()));

      tuner.SetArguments(definitionId, { argOut, argIn1, inSize, argIn2, in2N, isWithin });

      const auto &size = out.GetCorrelations().info.GetPaddedSize();
      tuner.SetLauncher(kernelId, [this, &size](ktt::ComputeInterface &interface) {
        auto blockDim = interface.GetCurrentLocalSize(definitionId);
        ktt::DimensionVector gridDim(size.x, size.y, size.z);
        gridDim.RoundUp(blockDim);
        gridDim.Divide(blockDim);
        interface.RunKernelAsync(definitionId, interface.GetAllQueues().at(0), gridDim, blockDim);
      });

      if (ShouldTune()) {
        tuner.TuneIteration(kernelId, {});
      } else {
        // TODO GetBestConfiguration can be used once the KTT is able to synchronize
        // the best configuration from multiple KTT instances, or loads the best
        // configuration from previous runs
        // auto bestConfig = tuner.GetBestConfiguration(kernelId);
        auto bestConfig = tuner.CreateConfiguration(kernelId,
          { { "blockSizeX", static_cast<uint64_t>(32) },
            { "blockSizeY", static_cast<uint64_t>(32) },
            { "TILE", static_cast<uint64_t>(8) } });
        tuner.Run(kernelId, bestConfig, {});// run is blocking call
        // arguments shall be removed once the run is done
      }
      return true;
    };
  };
}// namespace

void Correlation_CUDA::Synchronize() { GetHelper().GetTuner().Synchronize(); }

std::vector<std::unique_ptr<Correlation_CUDA::Strategy>> Correlation_CUDA::GetStrategies() const
{
  std::vector<std::unique_ptr<Correlation_CUDA::Strategy>> vec;
  vec.emplace_back(std::make_unique<Strategy1>(*this));
  return vec;
}

}// namespace umpalumpa::correlation
