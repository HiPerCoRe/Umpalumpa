#pragma once
#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>

namespace umpalumpa::fourier_processing {
class Settings
{
public:
  Settings(fourier_transformation::Locality loc) : locality(loc) {}

  bool IsEquivalentTo(const Settings &ref) const
  {
    return locality == ref.locality && center == ref.center && normalize == ref.normalize
           && applyFilter == ref.applyFilter;
  }

  auto GetLocality() const { return locality; }

  bool IsOutOfPlace() const { return locality == fourier_transformation::Locality::kOutOfPlace; }

  int GetVersion() const { return version; }

  void SetCenter(bool val) { this->center = val; }
  void SetNormalize(bool val) { this->normalize = val; }
  void SetApplyFilter(bool val) { this->applyFilter = val; }

  bool GetCenter() const { return center; }
  bool GetNormalize() const { return normalize; }
  bool GetApplyFilter() const { return applyFilter; }

private:
  static constexpr int version = 1;
  fourier_transformation::Locality locality;
  bool center = false;
  bool normalize = false;
  bool applyFilter = false;
};
}// namespace umpalumpa::fourier_processing
