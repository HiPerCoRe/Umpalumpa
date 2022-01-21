#include <libumpalumpa/algorithms/fourier_reconstruction/afr.hpp>
#include <libumpalumpa/math/kaiser.hpp>

namespace umpalumpa::fourier_reconstruction {

Constants AFR::CreateConstants(const AFR::InputData &in, const Settings &s)
{
  Constants c = {};
  // TODO review if these casts are necessary
  c.cMaxVolumeIndexX = static_cast<int>(in.GetVolume().info.GetSize().x - 1);
  c.cMaxVolumeIndexYZ = static_cast<int>(in.GetVolume().info.GetSize().y - 1);
  c.cBlobRadius = s.GetBlobRadius();
  c.cOneOverBlobRadiusSqr = 1.f / (s.GetBlobRadius() * s.GetBlobRadius());
  c.cBlobAlpha = s.GetAlpha();
  c.cIw0 = 1.f / math::KaiserFourierValue(0.f, s.GetBlobRadius(), s.GetAlpha(), s.GetBlobOrder());
  c.cIDeltaSqrt = float(in.GetBlobTable().info.Elems() - 1) / (s.GetBlobRadius() * s.GetBlobRadius());
  c.cOneOverBessiOrderAlpha = 1 / math::getBessiOrderAlpha(s.GetBlobOrder(), s.GetAlpha());
  return c;
}

// FIXME maybe return bool and check that the payload is valid and non-empty
void AFR::FillBlobTable(const AFR::InputData &in, const Settings &s)
{
  const auto &table = in.GetBlobTable();
  auto freq = s.GetBlobRadius() * std::sqrt(1.f / static_cast<float>(table.info.Elems() - 1));
  auto *ptr = reinterpret_cast<float *>(table.GetPtr());
  const auto c = CreateConstants(in, s);
  for (size_t i = 0; i < table.info.Elems(); ++i) {
    ptr[i] =
      math::KaiserValue(
        freq * static_cast<float>(std::sqrt(i)), s.GetBlobRadius(), s.GetAlpha(), s.GetBlobOrder())
      * c.cIw0;
  }
}

}// namespace umpalumpa::fourier_reconstruction