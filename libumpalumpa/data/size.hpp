#pragma once

#include <cstddef>
#include <libumpalumpa/data/dimensionality.hpp>

namespace umpalumpa {
namespace data {

  class Size
  {
  public:
    static inline Dimensionality GetDim(__attribute__((unused)) size_t x, size_t y, size_t z)
    {
      if ((z >= 2) && (y >= 2)) { return Dimensionality::k3Dim; }
      if ((z == 1) && (y >= 2)) { return Dimensionality::k2Dim; }
      return Dimensionality::k1Dim;
    }

    explicit Size(size_t xSize, size_t ySize, size_t zSize, size_t nSize)
      : x(xSize), y(ySize), z(zSize), n(nSize), dim(GetDim(xSize, ySize, zSize)), single(xSize * ySize * zSize),
        total(xSize * ySize * zSize * nSize)
    {}

    bool IsValid() const { return (0 != x) && (0 != y) && (0 != z) && (0 != n); }

    Size CopyFor(size_t newN) const { return Size(x, y, z, newN); }

    constexpr bool operator==(const Size &other) const
    {
      return (total == other.total) && (single == other.single) && (dim == other.dim) && (n == other.n)
             && (z == other.z) && (y == other.y) && (x == other.x);
    }
    constexpr bool operator!=(const Size &other) const { return !(*this == other); }
    constexpr bool operator<(const Size &other) const { return total < other.total; }
    constexpr bool operator>(const Size &other) const { return other < *this; }
    constexpr bool operator<=(const Size &other) const { return !(other < *this); }
    constexpr bool operator>=(const Size &other) const { return !(*this < other); }

    constexpr Dimensionality getDim() const { return dim; }

    const size_t x;
    const size_t y;
    const size_t z;
    const size_t n;
    const Dimensionality dim;
    const size_t single;
    const size_t total;
  };

}// namespace data
}// namespace umpalumpa
