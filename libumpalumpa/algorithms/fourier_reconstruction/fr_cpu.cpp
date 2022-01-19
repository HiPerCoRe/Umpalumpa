#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cpu_kernels.hpp>
#include <libumpalumpa/math/kaiser.hpp>
#include <libumpalumpa/math/bessier.hpp>
#include <libumpalumpa/utils/bool_expand.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space.hpp>

namespace umpalumpa::fourier_reconstruction {

namespace {// to avoid poluting
  struct Strategy1 final : public FRCPU::Strategy
  {
    // Inherit constructor
    using FRCPU::Strategy::Strategy;

    bool Init() override { return true; }

    std::string GetName() const override { return "Strategy1"; }

    auto CreateConstants(const AFR::InputData &in, const Settings &s)
    {
      Constants c = {};
      // TODO review if these casts are necessary
      c.cMaxVolumeIndexX = static_cast<int>(in.GetVolume().info.GetSize().x);
      c.cMaxVolumeIndexYZ = static_cast<int>(in.GetVolume().info.GetSize().y);
      c.cBlobRadius = s.GetBlobRadius();
      c.cOneOverBlobRadiusSqr = 1.f / (s.GetBlobRadius() * s.GetBlobRadius());
      c.cBlobAlpha = s.GetAlpha();
      c.cIw0 =
        1.f / math::KaiserFourierValue(0.f, s.GetBlobRadius(), s.GetAlpha(), s.GetBlobOrder());
      c.cIDeltaSqrt =
        0 /* FIXME BLOB_TABLE_SIZE_SQRT - 1 */ / (s.GetBlobRadius() * s.GetBlobRadius());
      c.cOneOverBessiOrderAlpha = 1 / math::getBessiOrderAlpha(s.GetBlobOrder(), s.GetAlpha());
      return c;
    }

    bool Execute(const AFR::OutputData &, const AFR::InputData &in) override
    {
      const auto &s = alg.GetSettings();

      float f;

      utils::ExpandBools<FR>::Expand(s.GetInterpolation() == Settings::Interpolation::kLookup,
        s.GetAlpha() <= 15.f,
        s.GetType() == Settings::Type::kFast,
        s.GetBlobOrder(),
        reinterpret_cast<std::complex<float> *>(in.GetVolume().GetPtr()),
        reinterpret_cast<float *>(in.GetWeight().GetPtr()),
        static_cast<int>(in.GetFFT().info.GetSize().x),// TODO maybe we can use unsigned?
        static_cast<int>(in.GetFFT().info.GetSize().y),// TODO maybe we can use unsigned?
        reinterpret_cast<std::complex<float> *>(in.GetFFT().GetPtr()),
        reinterpret_cast<TraverseSpace *>(in.GetTraverseSpace().GetPtr()),
        // // reinterpret_cast<TraverseSpace *>(in.GetSpace().GetPtr()),
        &f,// FIXME add
        CreateConstants(in, s));
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