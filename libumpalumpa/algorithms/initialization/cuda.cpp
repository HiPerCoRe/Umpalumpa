#include <libumpalumpa/algorithms/initialization/cuda.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>

namespace umpalumpa::initialization {

namespace {// to avoid poluting
  inline static const auto kKernelFile =
    utils::GetSourceFilePath("libumpalumpa/algorithms/initialization/cuda_kernels.cu");

  struct BasicInit final : public CUDA::KTTStrategy
  {
    // Inherit constructor
    using CUDA::KTTStrategy::KTTStrategy;

    size_t GetHash() const override { return 0; }

    bool IsSimilarTo(const TunableStrategy &) const override
    {
      // TODO real similarity check
      return false;
    }

    bool InitImpl() override
    {
      return !alg.GetInputRef().GetData().info.IsPadded()
             && alg.GetInputRef().GetValue().dataInfo.GetType().Is<float>();
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
