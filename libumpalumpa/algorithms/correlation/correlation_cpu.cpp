#include <libumpalumpa/algorithms/correlation/correlation_cpu.hpp>
#include <libumpalumpa/algorithms/correlation/correlation_cpu_kernels.hpp>

namespace umpalumpa::correlation {

namespace {// to avoid poluting

  struct Strategy1 final : public Correlation_CPU::Strategy
  {
    // Inherit constructor
    using Correlation_CPU::Strategy::Strategy;

    bool Init() override
    {
      // FIXME check settings
      const auto &out = alg.Get().GetOutputRef();
      const auto &in = alg.Get().GetInputRef();
      return ACorrelation::IsFloat(out, in);
    }

    std::string GetName() const override { return "Strategy1"; }

    bool Execute(const Correlation_CPU::OutputData &out,
      const Correlation_CPU::InputData &in) override
    {
      if (!in.GetData1().IsValid() || in.GetData1().IsEmpty()
          || !out.GetCorrelations().IsValid()// FIXME refactor
          || out.GetCorrelations().IsEmpty())
        return false;

      const auto &s = alg.GetSettings();
      if (s.GetCenter()) {
        if (in.GetData1().ptr == in.GetData2().ptr) {
          correlate2D<float, true, true>(
            reinterpret_cast<std::complex<float> *>(out.GetCorrelations().ptr),
            reinterpret_cast<std::complex<float> *>(in.GetData1().ptr),
            in.GetData1().info.GetSize(),
            reinterpret_cast<std::complex<float> *>(in.GetData2().ptr),
            in.GetData2().info.GetSize().n);
        } else {
          correlate2D<float, true, false>(
            reinterpret_cast<std::complex<float> *>(out.GetCorrelations().ptr),
            reinterpret_cast<std::complex<float> *>(in.GetData1().ptr),
            in.GetData1().info.GetSize(),
            reinterpret_cast<std::complex<float> *>(in.GetData2().ptr),
            in.GetData2().info.GetSize().n);
        }
      } else {
        if (in.GetData1().ptr == in.GetData2().ptr) {
          correlate2D<float, false, true>(
            reinterpret_cast<std::complex<float> *>(out.GetCorrelations().ptr),
            reinterpret_cast<std::complex<float> *>(in.GetData1().ptr),
            in.GetData1().info.GetSize(),
            reinterpret_cast<std::complex<float> *>(in.GetData2().ptr),
            in.GetData2().info.GetSize().n);
        } else {
          correlate2D<float, false, false>(
            reinterpret_cast<std::complex<float> *>(out.GetCorrelations().ptr),
            reinterpret_cast<std::complex<float> *>(in.GetData1().ptr),
            in.GetData1().info.GetSize(),
            reinterpret_cast<std::complex<float> *>(in.GetData2().ptr),
            in.GetData2().info.GetSize().n);
        }
      }
      return true;
    };
  };
}// namespace

std::vector<std::unique_ptr<Correlation_CPU::Strategy>> Correlation_CPU::GetStrategies() const
{
  std::vector<std::unique_ptr<Correlation_CPU::Strategy>> vec;
  vec.emplace_back(std::make_unique<Strategy1>(*this));
  return vec;
}

}// namespace umpalumpa::correlation
