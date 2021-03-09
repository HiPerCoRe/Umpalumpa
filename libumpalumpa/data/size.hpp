#ifndef LIBUMPALUMPA_DATA_SIZE
#define LIBUMPALUMPA_DATA_SIZE
#include <cstddef>
#include <libumpalumpa/data/dimensionality.hpp>

namespace umpalumpa {
namespace data {

class Size {
   public:
    static inline Dimensionality FromSize(__attribute__((unused)) size_t x, size_t y, size_t z) {
        if ((z >= 2) && (y >= 2)) {
            return Dimensionality::k3Dim;
        }
        if ((z == 1) && (y >= 2)) {
            return Dimensionality::k2Dim;
        }
        return Dimensionality::k1Dim;
    }

    explicit Size(size_t xSize, size_t ySize, size_t zSize, size_t nSize)
        : x(xSize),
          y(ySize),
          z(zSize),
          n(nSize),
          dim(FromSize(xSize, ySize, zSize)),
          single(xSize * ySize * zSize),
          total(xSize * ySize * zSize * nSize) {}

    bool IsValid() const { return (0 != x) && (0 != y) && (0 != z) && (0 != n); }

    const size_t x;
    const size_t y;
    const size_t z;
    const size_t n;
    const Dimensionality dim;
    const size_t single;
    const size_t total;
};

}  // namespace data
}  // namespace umpalumpa
#endif /* LIBUMPALUMPA_DATA_SIZE */
