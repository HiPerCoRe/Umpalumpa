#include <libumpalumpa/algorithms/initialization/cuda.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>

namespace umpalumpa::initialization {

namespace {// to avoid poluting
  inline static const auto kKernelFile =
    utils::GetSourceFilePath("libumpalumpa/algorithms/initialization/cuda_kernels.cu");

  struct BasicInit : public CUDA::KTTStrategy
  {
    // Inherit constructor
    using CUDA::KTTStrategy::KTTStrategy;

    static constexpr auto kInitialize = "Initialize";

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

    bool IsEqualTo(const TunableStrategy &) const override { return true; }

    bool IsSimilarTo(const TunableStrategy &) const override { return true; }

    bool InitImpl() override
    {
      const auto &in = alg.Get().GetInputRef();
      bool canProcess = !in.GetData().info.IsPadded()
                        && alg.GetInputRef().GetValue().dataInfo.GetType().Is<float>();
      if (!canProcess) return false;

      auto &tuner = kttHelper.GetTuner();

      // ensure that we have the kernel loaded to KTT
      // this has to be done in critical section, as multiple instances of this algorithm
      // might run on the same worker
      const auto &size = in.GetData().info.GetSize();
      std::lock_guard<std::mutex> lck(kttHelper.GetMutex());
      AddKernelDefinition(
        kInitialize, kKernelFile, ktt::DimensionVector{ size.total }, { "float" });
      auto definitionId = GetDefinitionId();

      AddKernel(kInitialize, definitionId);


      auto kernelId = GetKernelId();
      tuner.AddParameter(
        kernelId, "blockSize", std::vector<uint64_t>{ 32, 64, 128, 256, 512, 1024 });
      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Local,
        ktt::ModifierDimension::X,
        "blockSize",
        ktt::ModifierAction::Multiply);
      tuner.SetSearcher(kernelId, std::make_unique<ktt::RandomSearcher>());
      return canProcess;
    }

    std::string GetName() const override { return "BasicInit"; }

    bool Execute(const Abstract::OutputData &, const Abstract::InputData &in) override
    {
      auto IsFine = [](const auto &p) { return p.IsValid() && !p.IsEmpty(); };
      if (!IsFine(in.GetData()) || !IsFine(in.GetValue())) return false;

      if (0.f == reinterpret_cast<float *>(in.GetValue().GetPtr())[0]) {
        const auto &d = in.GetData();
        // FIXME use proper stream
        CudaErrchk(cudaMemsetAsync(d.GetPtr(), 0, d.GetRequiredBytes()));
        return true;
      }
      return false;
    };
  };
}// namespace

void CUDA::Synchronize()
{
  // FIXME synchronize correct stream only
  CudaErrchk(cudaStreamSynchronize(0));
}

std::vector<std::unique_ptr<CUDA::Strategy>> CUDA::GetStrategies() const
{
  std::vector<std::unique_ptr<CUDA::Strategy>> vec;
  vec.emplace_back(std::make_unique<BasicInit>(*this));
  return vec;
}

}// namespace umpalumpa::initialization
