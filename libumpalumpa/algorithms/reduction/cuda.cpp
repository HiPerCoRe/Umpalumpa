#include <libumpalumpa/algorithms/reduction/cuda.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>

namespace umpalumpa::reduction {

namespace {// to avoid poluting
  inline static const auto kKernelFile =
    utils::GetSourceFilePath("libumpalumpa/algorithms/reduction/cuda_kernels.cu");

  struct PiecewiseSum final : public CUDA::KTTStrategy
  {
    // Inherit constructor
    using CUDA::KTTStrategy::KTTStrategy;

    static constexpr auto kKernel = "PiecewiseSum";

    size_t GetHash() const override { return 0; }
    bool IsSimilarTo(const TunableStrategy &) const override
    {
      // TODO real similarity check
      return false;
    }

    bool InitImpl() override
    {
      const auto &in = alg.GetInputRef();
      bool isValidOp = alg.Get().GetSettings().GetOperation() == Settings::Operation::kPiecewiseSum;
      bool isFloat = in.GetData().dataInfo.GetType().Is<float>();
      bool canProcess = isValidOp && isFloat && !in.GetData().info.IsPadded()
                        && !alg.GetOutputRef().GetData().info.IsPadded();

      if (!canProcess) return false;

      auto &size = in.GetData().info.GetSize();
      auto &tuner = kttHelper.GetTuner();

      // ensure that we have the kernel loaded to KTT
      // this has to be done in critical section, as multiple instances of this algorithm
      // might run on the same worker
      std::lock_guard<std::mutex> lck(kttHelper.GetMutex());
      AddKernelDefinition(kKernel, kKernelFile, ktt::DimensionVector{ size.total });
      auto definitionId = GetDefinitionId();

      AddKernel(kKernel, definitionId);
      auto kernelId = GetKernelId();

      tuner.AddParameter(
        kernelId, "blockSize", std::vector<uint64_t>{ 32, 64, 128, 256, 512, 1024 });

      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Local,
        ktt::ModifierDimension::X,
        "blockSize",
        ktt::ModifierAction::Multiply);
      return true;
    }

    std::string GetName() const override { return "PiecewiseSum"; }

    bool Execute(const Abstract::OutputData &out, const Abstract::InputData &in) override
    {
      auto IsFine = [](const auto &p) { return p.IsValid() && !p.IsEmpty(); };
      if (!IsFine(in.GetData()) || !IsFine(out.GetData())) return false;

      // prepare input data
      auto &tuner = kttHelper.GetTuner();
      auto argIn = AddArgumentVector<float>(in.GetData(), ktt::ArgumentAccessType::ReadOnly);
      auto argSize = tuner.AddArgumentScalar(in.GetData().info.GetSize());

      // prepare output data
      auto argOut = AddArgumentVector<float>(out.GetData(), ktt::ArgumentAccessType::ReadWrite);

      tuner.SetArguments(GetDefinitionId(), { argOut, argIn, argSize });

      auto &size = in.GetData().info.GetSize();
      tuner.SetLauncher(GetKernelId(), [this, &size](ktt::ComputeInterface &interface) {
        auto blockDim = interface.GetCurrentLocalSize(GetKernelId());
        ktt::DimensionVector gridDim(size.total);
        gridDim.RoundUp(blockDim);
        gridDim.Divide(blockDim);
        interface.RunKernelAsync(GetKernelId(), interface.GetAllQueues().at(0), gridDim, blockDim);
      });

      if (ShouldTune()) {
        tuner.TuneIteration(GetKernelId(), {});
      } else {
        // TODO GetBestConfiguration can be used once the KTT is able to synchronize
        // the best configuration from multiple KTT instances, or loads the best
        // configuration from previous runs
        // auto bestConfig = tuner.GetBestConfiguration(kernelId);
        auto bestConfig = tuner.CreateConfiguration(
          GetKernelId(), { { "blockSize", static_cast<uint64_t>(1024) } });
        tuner.Run(GetKernelId(), bestConfig, {});// run is blocking call
      }
      // arguments shall be removed once the run is done

      return true;
    };
  };
}// namespace

void CUDA::Synchronize() { GetHelper().GetTuner().Synchronize(); }

std::vector<std::unique_ptr<CUDA::Strategy>> CUDA::GetStrategies() const
{
  std::vector<std::unique_ptr<CUDA::Strategy>> vec;
  vec.emplace_back(std::make_unique<PiecewiseSum>(*this));
  return vec;
}

}// namespace umpalumpa::reduction
