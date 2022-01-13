#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cpu_kernels.hpp>
#include <libumpalumpa/utils/bool_expand.hpp>

namespace umpalumpa::fourier_reconstruction {

namespace {// to avoid poluting
  struct Strategy1 final : public FRCPU::Strategy
  {
    // Inherit constructor
    using FRCPU::Strategy::Strategy;

    bool Init() override { return true; }

    std::string GetName() const override { return "Strategy1"; }

    bool Execute(const AFR::OutputData &, const AFR::InputData &) override
    {
      const auto &s = alg.GetSettings();

      utils::ExpandBools<FR>::Expand(
        s.GetInterpolation() == Settings::Interpolation::kLookup, s.GetBlobOrder());

      return true;
    }
  };
}// namespace


std::vector<std::unique_ptr<FRCPU::Strategy>> FRCPU::GetStrategies() const
{
  std::vector<std::unique_ptr<FRCPU::Strategy>> vec;
  vec.emplace_back(std::make_unique<Strategy1>(*this));
  return vec;
}
}// namespace umpalumpa::fourier_reconstruction