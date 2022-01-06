#include <complex>
#include <functional>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu_kernels.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/utils/bool_expand.hpp>

namespace umpalumpa::fourier_processing {

namespace {// to avoid poluting
  struct Strategy1 final : public FPCPU::Strategy
  {
    // Inherit constructor
    using FPCPU::Strategy::Strategy;

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
      utils::ExpandBools<ScaleFFT2DCPU>::Expand(s.GetApplyFilter(),
        s.GetNormalize(),
        s.GetCenter(),
        s.GetMaxFreq().has_value(),
        s.GetShift(),
        reinterpret_cast<std::complex<float> *>(in.GetData().GetPtr()),
        reinterpret_cast<std::complex<float> *>(out.GetData().GetPtr()),
        in.GetData().info.GetSize(),
        in.GetData().info.GetSpatialSize(),
        out.GetData().info.GetSize(),
        reinterpret_cast<float *>(in.GetFilter().GetPtr()),
        1.f / static_cast<float>(in.GetData().info.GetPaddedSpatialSize().single),
        s.GetMaxFreq().value_or(0));
      return true;
    }
  };
}// namespace

std::vector<std::unique_ptr<FPCPU::Strategy>> FPCPU::GetStrategies() const
{
  std::vector<std::unique_ptr<FPCPU::Strategy>> vec;
  vec.emplace_back(std::make_unique<Strategy1>(*this));
  return vec;
}
}// namespace umpalumpa::fourier_processing
