#include <complex>
#include <functional>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu_kernels.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa {
namespace fourier_processing {

  namespace {// to avoid poluting
    struct Strategy1 : public FP_CPU::Strategy
    {
      static constexpr auto kStrategyName = "Strategy1";

      bool
        Init(const AFP::OutputData &out, const AFP::InputData &in, const Settings &) override final
      {
        // FIXME check settings
        return (in.data.dataInfo.type == data::DataType::kComplexFloat)
               && (in.filter.dataInfo.type == data::DataType::kFloat)
               && (out.data.dataInfo.type == data::DataType::kComplexFloat);
      }

      std::string GetName() const override final { return kStrategyName; }

      bool Execute(const AFP::OutputData &out,
        const AFP::InputData &in,
        const Settings &s) override final
      {
        if (!in.data.IsValid() || in.data.IsEmpty() || !out.data.IsValid() || out.data.IsEmpty())
          return false;
        scaleFFT2DCPU(reinterpret_cast<std::complex<float> *>(in.data.ptr),
          reinterpret_cast<std::complex<float> *>(out.data.ptr),
          in.data.info.GetSize(),
          out.data.info.GetSize(),
          reinterpret_cast<float *>(in.filter.ptr),
          1.f / static_cast<float>(in.data.info.GetPaddedSpatialSize().single),
          s.GetApplyFilter(),
          s.GetNormalize(),
          s.GetCenter());
        return true;
      }
    };
  }// namespace

  bool FP_CPU::Init(const OutputData &out, const InputData &in, const Settings &s)
  {
    SetSettings(s);
    auto tryToAdd = [this, &out, &in, &s](auto i) {
      bool canAdd = i->Init(out, in, s);
      if (canAdd) {
        spdlog::debug("Found valid strategy {}", i->GetName());
        strategy = std::move(i);
      }
      return canAdd;
    };

    return tryToAdd(std::make_unique<Strategy1>());
  }

  bool FP_CPU::Execute(const OutputData &out, const InputData &in)
  {
    if (!this->IsValid(out, in)) return false;
    return strategy->Execute(out, in, GetSettings());
  }
}// namespace fourier_processing
}// namespace umpalumpa

