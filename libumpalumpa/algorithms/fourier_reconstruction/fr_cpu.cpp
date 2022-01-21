#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cpu_kernels.hpp>
#include <libumpalumpa/utils/bool_expand.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space.hpp>

namespace umpalumpa::fourier_reconstruction {

namespace {// to avoid poluting
  struct Strategy1 final : public FRCPU::Strategy
  {
    // Inherit constructor
    using FRCPU::Strategy::Strategy;

    bool Init() override
    {
      const auto &in = alg.Get().GetInputRef();
      // we can process only odd sized data
      if (0 == in.GetVolume().info.GetSize().x % 2) { return false; }
      return true;
    }

    std::string GetName() const override { return "Strategy1"; }

    bool Execute(const AFR::OutputData &, const AFR::InputData &in) override
    {
      const auto &s = alg.GetSettings();
      if (s.GetInterpolation() == Settings::Interpolation::kLookup && in.GetBlobTable().IsEmpty()) {
        return false;
      }
      utils::ExpandBools<FR>::Expand(s.GetType() == Settings::Type::kFast,
        s.GetAlpha() <= 15.f,
        s.GetInterpolation() == Settings::Interpolation::kLookup,
        s.GetBlobOrder(),
        reinterpret_cast<std::complex<float> *>(in.GetVolume().GetPtr()),
        reinterpret_cast<float *>(in.GetWeight().GetPtr()),
        static_cast<int>(in.GetFFT().info.GetSize().x),// TODO maybe we can use unsigned?
        static_cast<int>(in.GetFFT().info.GetSize().y),// TODO maybe we can use unsigned?
        reinterpret_cast<std::complex<float> *>(in.GetFFT().GetPtr()),
        reinterpret_cast<TraverseSpace *>(in.GetTraverseSpace().GetPtr()),
        reinterpret_cast<float *>(in.GetBlobTable().GetPtr()),
        AFR::CreateConstants(in, s));
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