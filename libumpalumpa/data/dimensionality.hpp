#pragma once

namespace umpalumpa {
namespace data {

  enum class Dimensionality {
    kInvalid = 0,
    k1Dim = 1,
    k2Dim = 2,
    k3Dim = 3,
  };

  [[maybe_unused]]
  static int ToInt(Dimensionality d) {
    switch (d) {
      case Dimensionality::k1Dim: return 1;
      case Dimensionality::k2Dim: return 2;
      case Dimensionality::k3Dim: return 3;
      default: return 0;
    }
  }
}// namespace data
}// namespace umpalumpa
