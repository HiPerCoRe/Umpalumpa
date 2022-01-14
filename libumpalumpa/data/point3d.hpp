#pragma once

#include <libumpalumpa/utils/cuda_compatibility.hpp>

namespace umpalumpa {// must be two namespaces for compatiblity with CUDA
namespace data {

  template<typename T> class Point3D
  {
  public:
    T x = 0;
    T y = 0;
    T z = 0;

    CUDA_HD
    Point3D &operator/=(const T &rhs) const { return Point3D(x / rhs, y / rhs, z / rhs); }

    CUDA_HD
    friend Point3D operator/(const Point3D &lhs, T rhs) { return lhs /= rhs; }
  };

}// namespace data
}// namespace umpalumpa