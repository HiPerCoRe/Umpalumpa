#include <libumpalumpa/algorithms/reduction/cuda.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>

namespace umpalumpa::reduction {

namespace {// to avoid poluting
  inline static const auto kKernelFile =
    utils::GetSourceFilePath("libumpalumpa/algorithms/reduction/cuda_kernels.cu");

  struct PiecewiseSum : public CUDA::KTTStrategy
  {
    // Inherit constructor
    using CUDA::KTTStrategy::KTTStrategy;

    static constexpr auto kKernel = "PiecewiseSum";

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
      return { kttHelper.GetTuner().CreateConfiguration(
        GetKernelId(), { { "blockSize", static_cast<uint64_t>(1024) } }) };
    }

    bool IsEqualTo(const TunableStrategy &ref) const override
    {
      bool isEqual = true;
      try {
        auto &refStrat = dynamic_cast<const PiecewiseSum &>(ref);
        isEqual = isEqual && GetOutputRef().IsEquivalentTo(refStrat.GetOutputRef());
        isEqual = isEqual && GetInputRef().IsEquivalentTo(refStrat.GetInputRef());
        isEqual = isEqual && GetSettings().IsEquivalentTo(refStrat.GetSettings());
        // Additional checks might be needed
      } catch (std::bad_cast &) {
        isEqual = false;
      }
      return isEqual;
    }

    bool IsSimilarTo(const TunableStrategy &ref) const override
    {
      bool isSimilar = true;
      try {
        auto &refStrat = dynamic_cast<const PiecewiseSum &>(ref);
        isSimilar = isSimilar && GetOutputRef().IsEquivalentTo(refStrat.GetOutputRef());
        isSimilar = isSimilar && GetInputRef().IsEquivalentTo(refStrat.GetInputRef());
        isSimilar = isSimilar && GetSettings().IsEquivalentTo(refStrat.GetSettings());
        // Using naive similarity: same as equality
      } catch (std::bad_cast &) {
        isSimilar = false;
      }
      return isSimilar;
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

      SetArguments(GetDefinitionId(), { argOut, argIn, argSize });

      auto &size = in.GetData().info.GetSize();
      tuner.SetLauncher(GetKernelId(), [this, &size](ktt::ComputeInterface &interface) {
        auto blockDim = interface.GetCurrentLocalSize(GetKernelId());
        ktt::DimensionVector gridDim(size.total);
        gridDim.RoundUp(blockDim);
        gridDim.Divide(blockDim);
        if (ShouldBeTuned(GetKernelId())) {
          interface.RunKernel(GetKernelId(), gridDim, blockDim);
        } else {
          WaitBeforeDestruction(interface.RunKernelAsync(
            GetKernelId(), interface.GetAllQueues().at(0), gridDim, blockDim));
        }
      });

      ExecuteKernel(GetKernelId());

      return true;
    };
  };
}// namespace

void CUDA::Synchronize() { GetHelper().GetTuner().SynchronizeDevice(); }

std::vector<std::unique_ptr<CUDA::Strategy>> CUDA::GetStrategies() const
{
  std::vector<std::unique_ptr<CUDA::Strategy>> vec;
  vec.emplace_back(std::make_unique<PiecewiseSum>(*this));
  return vec;
}

}// namespace umpalumpa::reduction
