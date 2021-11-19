#pragma once

#include <libumpalumpa/data/size.hpp>
#include <limits>

namespace umpalumpa::extrema_finder {

/**
 * Find location or value of the extrema defined by the comparator.
 * Data has to contain at least one (1) value.
 * Returned location is whole number.
 * For X-dimensional data, X-dimensional location is expected, i.e.
 * returns 2D location for 2D data. The order on Nth position is [X, [Y], [Z]]
 * Data has to be copyable.
 *
 * All checks are expected to be done by caller
 **/
template<bool Values, bool Locations, typename T, typename C>
bool FindSingleExtremaCPU(T *__restrict__ vals,
  float *__restrict__ locs,
  T *const __restrict__ data,
  const umpalumpa::data::Size &size,
  const C &comp)
{
  using umpalumpa::data::Dimensionality;
  for (size_t n = 0; n < size.n; ++n) {
    const size_t offset = n * size.single;
    auto extrema = data[offset];
    size_t location = 0;
    for (size_t i = 1; i < size.single; ++i) {
      auto &v = data[offset + i];
      if (comp(v, extrema)) {
        location = i;
        extrema = v;
      }
    }
    // save location
    if (Locations) {
      auto *dest = locs + n * size.GetDimAsNumber();
      switch (size.GetDim()) {
      case Dimensionality::k1Dim:
        dest[0] = static_cast<float>(location);
        break;
      case Dimensionality::k2Dim: {
        size_t y = location / size.x;
        size_t x = location % size.x;
        dest[0] = static_cast<float>(x);
        dest[1] = static_cast<float>(y);
      } break;
      case Dimensionality::k3Dim: {
        size_t z = location / (size.x * size.y);
        size_t tmp = location % (size.x * size.y);
        size_t y = tmp / size.x;
        size_t x = tmp % size.x;
        dest[0] = static_cast<float>(x);
        dest[1] = static_cast<float>(y);
        dest[2] = static_cast<float>(z);
      } break;
      default:
        locs[n] = std::numeric_limits<float>::quiet_NaN();
      }
    }
    // save value
    if (Values) { vals[n] = extrema; }
  }
  return true;
}

template<bool Values, bool Locations, typename T, typename C>
bool FindSingleExtremaInRectangle2DCPU(T *__restrict__ vals,
  T *__restrict__ locs,
  T *const __restrict__ data,
  size_t offsetX,
  size_t offsetY,
  const umpalumpa::data::Size &rectSize,
  const umpalumpa::data::Size &totalSize,
  const C &comp)
{
  // all checks are expected to be done by caller
  for (size_t n = 0; n < totalSize.n; ++n) {
    const size_t offset = n * totalSize.single;
    auto rectStart = offsetY * totalSize.x + offsetX;
    auto extrema = data[offset + rectStart];
    auto extremaLoc = offset + rectStart;
    for (size_t i = rectStart; i < rectStart + rectSize.y * totalSize.x; i += totalSize.x) {
      for (size_t j = i; j < i + rectSize.x; j++) {
        auto &v = data[offset + j];
        if (comp(v, extrema)) {
          extrema = v;
          extremaLoc = j;
        }
      }
    }
    if (Locations) { locs[n] = static_cast<float>(extremaLoc); }
    if (Values) { vals[n] = static_cast<float>(extrema); }
  }
  return true;
}
}// namespace umpalumpa::extrema_finder
