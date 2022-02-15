#pragma once
#include <optional>
#include <cassert>
#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>
#include <iostream>

namespace umpalumpa::fourier_processing {
class Settings
{
public:
  Settings(fourier_transformation::Locality loc) : locality(loc) {}

  bool IsEquivalentTo(const Settings &ref) const
  {
    return locality == ref.locality && center == ref.center && normalize == ref.normalize
           && applyFilter == ref.applyFilter && shift == ref.shift && maxFreq == ref.maxFreq;
  }

  bool IsOutOfPlace() const { return locality == fourier_transformation::Locality::kOutOfPlace; }

  int GetVersion() const { return version; }

  void SetCenter(bool val) { this->center = val; }
  /**
   * Move low frequencies from corners to center or vice versa.
   * This is different from Centering, which centers FFT in the spatial domain.
   **/
  void SetShift(bool val) { this->shift = val; }
  void SetNormalize(bool val) { this->normalize = val; }
  void SetApplyFilter(bool val) { this->applyFilter = val; }
  /**
   * Set max norm (absolute square), in frequency, i.e. from [0, 1/2]
   **/
  void SetMaxFreq(float f)
  {
    assert(0 <= f);
    assert(f <= 0.5f);
    this->maxFreq = std::make_optional(f);
  }

  bool GetCenter() const { return center; }
  bool GetShift() const { return shift; }
  auto GetLocality() const { return locality; }
  bool GetNormalize() const { return normalize; }
  bool GetApplyFilter() const { return applyFilter; }
  auto GetMaxFreq() const { return maxFreq; }

  void Serialize(std::ostream &out) const
  {
    out << static_cast<int>(locality) << ' ' << center << ' ' << normalize << ' ' << applyFilter
        << ' ' << shift;
    out << ' ' << maxFreq.has_value();
    if (maxFreq.has_value()) { out << ' ' << maxFreq.value(); }
    out << '\n';
  }
  static auto Deserialize(std::istream &in)
  {
    int locality;
    bool center, normalize, applyFilter, shift, hasMaxFreq;
    in >> locality >> center >> normalize >> applyFilter >> shift >> hasMaxFreq;
    Settings s(static_cast<fourier_transformation::Locality>(locality));
    s.SetCenter(center);
    s.SetNormalize(normalize);
    s.SetApplyFilter(applyFilter);
    s.SetShift(shift);
    if (hasMaxFreq) {
      float maxFreq;
      in >> maxFreq;
      s.SetMaxFreq(maxFreq);
    }
    return s;
  }

private:
  static constexpr int version = 1;
  fourier_transformation::Locality locality;
  bool center = false;
  bool normalize = false;
  bool applyFilter = false;
  bool shift = false;
  std::optional<float> maxFreq;
};
}// namespace umpalumpa::fourier_processing
