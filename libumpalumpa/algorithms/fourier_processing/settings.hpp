#pragma once
#include <optional>
#include <cassert>
#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>

namespace umpalumpa::fourier_processing {
class Settings
{
public:
  Settings(fourier_transformation::Locality loc) : locality(loc) {}

  bool IsOutOfPlace() const { return locality == fourier_transformation::Locality::kOutOfPlace; }

  int GetVersion() const { return version; }

  void SetCenter(bool val) { this->center = val; }
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
  auto GetLocality() const { return locality; }
  bool GetNormalize() const { return normalize; }
  bool GetApplyFilter() const { return applyFilter; }
  auto GetMaxFreq() const { return maxFreq; }

private:
  static constexpr int version = 1;
  fourier_transformation::Locality locality;
  bool center = false;
  bool normalize = false;
  bool applyFilter = false;
  std::optional<float> maxFreq;
};
}// namespace umpalumpa::fourier_processing
