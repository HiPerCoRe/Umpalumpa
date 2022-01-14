#pragma once

#include <libumpalumpa/utils/cuda_compatibility.hpp>
#include <libumpalumpa/data/point3d.hpp>

namespace umpalumpa {

/** Do 3x3 x 1x3 matrix-vector multiplication */
CUDA_HD
template<typename T> void multiply(const T transform[3][3], data::Point3D<T> &inOut)
{
  T tmp0 = transform[0][0] * inOut.x + transform[0][1] * inOut.y + transform[0][2] * inOut.z;
  T tmp1 = transform[1][0] * inOut.x + transform[1][1] * inOut.y + transform[1][2] * inOut.z;
  T tmp2 = transform[2][0] * inOut.x + transform[2][1] * inOut.y + transform[2][2] * inOut.z;
  inOut.x = tmp0;
  inOut.y = tmp1;
  inOut.z = tmp2;
}
}