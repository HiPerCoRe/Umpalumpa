#include <libumpalumpa/operations/initialization/cuda.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>

namespace umpalumpa::initialization {

namespace {// to avoid poluting
  inline static const auto kKernelFile =
    utils::GetSourceFilePath("libumpalumpa/operations/initialization/cuda_kernels.cu");

  struct BasicInit : public CUDA::KTTStrategy
  {
    // Inherit constructor
    using CUDA::KTTStrategy::KTTStrategy;

    size_t GetHash() const override { return 0; }

    std::unique_ptr<tuning::Leader> CreateLeader() const override
    {
      return tuning::StrategyGroup::CreateLeader(*this, op);
    }
    tuning::StrategyGroup LoadTuningData() const override
    {
      return tuning::StrategyGroup::LoadTuningData(*this, op);
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
      return !op.GetInputRef().GetData().info.IsPadded()
             && op.GetInputRef().GetValue().dataInfo.GetType().Is<float>();
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
