#include <complex>
#include <functional>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu_kernels.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::fourier_processing {

namespace {// to avoid poluting
  struct Strategy1 final : public FP_CPU::Strategy
  {
    // Inherit constructor
    using FP_CPU::Strategy::Strategy;

    bool Init() override
    {
      // FIXME check settings
      const auto &out = alg.Get().GetOutputRef();
      const auto &in = alg.Get().GetInputRef();
      return AFP::IsFloat(out, in);
    }

    std::string GetName() const override { return "Strategy1"; }

    bool Execute(const AFP::OutputData &out, const AFP::InputData &in) override
    {
      if (!in.GetData().IsValid() || in.GetData().IsEmpty() || !out.GetData().IsValid()
          || out.GetData().IsEmpty())
        return false;

      const auto &s = alg.GetSettings();
      scaleFFT2DCPU(reinterpret_cast<std::complex<float> *>(in.GetData().ptr),
        reinterpret_cast<std::complex<float> *>(out.GetData().ptr),
        in.GetData().info.GetSize(),
        out.GetData().info.GetSize(),
        reinterpret_cast<float *>(in.GetFilter().ptr),
        1.f / static_cast<float>(in.GetData().info.GetPaddedSpatialSize().single),
        s.GetApplyFilter(),
        s.GetNormalize(),
        s.GetCenter());
      return true;
    }
  };
}// namespace

std::vector<std::unique_ptr<FP_CPU::Strategy>> FP_CPU::GetStrategies() const
{
  std::vector<std::unique_ptr<FP_CPU::Strategy>> vec;
  vec.emplace_back(std::make_unique<Strategy1>(*this));
  return vec;
}
}// namespace umpalumpa::fourier_processing
