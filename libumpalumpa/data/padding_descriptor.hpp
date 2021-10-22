#pragma once

// NVRTC can't handle std includes. <cstddef> is included to define size_t;
// however, NVRTC has size_t predefined so no include is needed
#ifndef __CUDACC_RTC__
#include <cstddef>
#endif

namespace umpalumpa {
namespace data {
  /**
   * Class describing number of elements used for padding.
   * If all sizes are equal to 0, no padding is used.
   * First character (X, Y, Z) is for dimension.
   * (Beg, End) defines padding at the beginning / at the end of data
   * Endxample: xBeg = 2; xEnd = 3; means that in X dimension,
   * the first two elemenst and the last three elements are padding.
   **/
  class PaddingDescriptor
  {
  public:
    /**
     * No padding
     **/
    PaddingDescriptor() : xBeg(0), xEnd(0), yBeg(0), yEnd(0), zBeg(0), zEnd(0){};

    PaddingDescriptor(size_t xB, size_t xE, size_t yB, size_t yE, size_t zB, size_t zE)
      : xBeg(xB), xEnd(xE), yBeg(yB), yEnd(yE), zBeg(zB), zEnd(zE){};

    inline size_t GetXBeg() const { return xBeg; }
    inline size_t GetXEnd() const { return xEnd; }
    inline size_t GetYBeg() const { return yBeg; }
    inline size_t GetYEnd() const { return yEnd; }
    inline size_t GetZBeg() const { return zBeg; }
    inline size_t GetZEnd() const { return zEnd; }

  private:
    size_t xBeg;
    size_t xEnd;
    size_t yBeg;
    size_t yEnd;
    size_t zBeg;
    size_t zEnd;
  };
}// namespace data
}// namespace umpalumpa