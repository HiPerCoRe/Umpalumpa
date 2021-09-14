#pragma once
#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>

namespace umpalumpa {
namespace fourier_transformation {
  class Settings {
  public:
    Settings(Locality loc, Direction dir) : locality(loc), direction(dir) {}

    Locality GetLocality() const { return locality; }
    Direction GetDirection() const { return direction; }

    bool IsForward() const { return direction == Direction::kForward; }
    bool IsOutOfPlace() const { return locality == Locality::kOutOfPlace; }

    int GetVersion() const { return version; }

    Settings CreateInverse() const {
      return Settings(locality,
          direction == Direction::kForward ? Direction::kInverse : Direction::kForward);
    }

  private:
    static constexpr int version = 1;
    //umpalumpa::data::Size spatial;
    //umpalumpa::data::Size freq;
    Locality locality;
    Direction direction;
  };
}
}
