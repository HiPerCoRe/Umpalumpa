#pragma once

// NVRTC can't handle std includes. <cstddef> is included to define size_t;
// however, NVRTC has size_t predefined so no include is needed
#ifndef __CUDACC_RTC__
#include <cstddef>
#endif
#include <libumpalumpa/data/dimensionality.hpp>

namespace umpalumpa {// must be two namespaces for compatiblity with CUDA
namespace data {

  /**
   * Class representing size of some object in up to 3 dimensions [x, y, z], including number of
   * of those objects [n].
   * 'Unused' dimensions have size 1. As such, this class cannot represent 2D and 3D sizes with
   * collapsed dimensions. In other words,
   * valid 1D size has Z==1 and Y==1 and X>=1,
   * valid 2D size has Z==1 and Y>=2 and X>=2,
   * valid 3D size has Z>=2 and Y>=2 and X>=2.
   **/
  class Size
  {
  public:
    static inline Dimensionality GetDim(size_t x, size_t y, size_t z)
    {
      if ((z >= 2) && (y >= 2) && (x >= 2)) { return Dimensionality::k3Dim; }
      if ((z == 1) && (y >= 2) && (x >= 2)) { return Dimensionality::k2Dim; }
      if ((z == 1) && (y == 1) && (x >= 1)) { return Dimensionality::k1Dim; }
      return Dimensionality::kInvalid;
    }

    // FIXME ensure that nobody creates size like x=1 y=2 ... (either exception here or check in
    // IsValid() or both, preferably exception here)
    explicit Size(size_t xSize, size_t ySize, size_t zSize, size_t nSize)
      : x(xSize), y(ySize), z(zSize), n(nSize), dim(GetDim(xSize, ySize, zSize)),
        single(xSize * ySize * zSize), total(xSize * ySize * zSize * nSize)
    {}

    bool IsValid() const { return (1 <= n) && (Dimensionality::kInvalid != dim); }

    Size CopyFor(size_t newN) const { return Size(x, y, z, newN); }

    constexpr bool operator==(const Size &other) const
    {
      return (total == other.total) && (single == other.single) && (dim == other.dim)
             && (n == other.n) && (z == other.z) && (y == other.y) && (x == other.x);
    }
    constexpr bool operator!=(const Size &other) const { return !(*this == other); }
    constexpr bool operator<(const Size &other) const { return total < other.total; }
    constexpr bool operator>(const Size &other) const { return other < *this; }
    constexpr bool operator<=(const Size &other) const { return !(other < *this); }
    constexpr bool operator>=(const Size &other) const { return !(*this < other); }

    constexpr Dimensionality GetDim() const { return dim; }

    constexpr unsigned short GetDimAsNumber() const
    {
      switch (dim) {
      case Dimensionality::k1Dim:
        return 1;
        break;
      case Dimensionality::k2Dim:
        return 2;
      case Dimensionality::k3Dim:
        return 3;
      default:
        return static_cast<unsigned short>(-1);// unsupported
      }
    }

    /**
     * Returns true if all sizes of this are the same as those of referenec except for N,
     * which can be lower or equal than refernce.N.
     **/
    constexpr bool IsEquivalentTo(const Size &ref) const
    {
      return (x == ref.x) && (y == ref.y) && (z == ref.z) && (n <= ref.n);
    }

    // these should be private + getters
    size_t x;
    size_t y;
    size_t z;
    size_t n;
    Dimensionality dim;
    size_t single;
    size_t total;
  };

}// namespace data
}// namespace umpalumpa
