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
        dest[n] = std::numeric_limits<float>::quiet_NaN();
      }
    }
    // save value
    if (Values) { vals[n] = extrema; }
  }
  return true;
}

/**
 * Find sub-pixel location or value of the extrema.
 * Data has to contain at least one (1) value.
 * Returned location is calculated by relative weigting in the given
 * window using the value contribution. Should the window reach behind the boundaries, those
 * values will be ignored. Only odd sizes of the window are valid.
 *
 * All checks are expected to be done by caller
 **/
template<typename T, unsigned WINDOW>
bool RefineLocation(float *__restrict__ locs,
  T *const __restrict__ data,
  const umpalumpa::data::Size &size)
{
  assert(WINDOW % 2 == 1);
  using umpalumpa::data::Dimensionality;
  auto half = (WINDOW - 1) / 2;
  const auto dim = size.GetDimAsNumber();
  if ((dim > 0) && (dim <= 3)) {
    // auto tmp = std::make_unique<T>(new LocWeight[WINDOW]);
    for (size_t n = 0; n < size.n; ++n) {
      auto *ptrLoc = locs + n * size.GetDimAsNumber();
      auto *ptr = data + n * size.single;
      auto refX = static_cast<size_t>(ptrLoc[0]);
      auto refY = (size.GetDimAsNumber() > 1) ? static_cast<size_t>(ptrLoc[1]) : 0;
      auto refZ = (size.GetDimAsNumber() > 2) ? static_cast<size_t>(ptrLoc[2]) : 0;
      auto refVal = data[n * size.single + refZ * size.x * size.y + refY * size.x + refX];
      // careful with unsigned operations
      auto startX = (half > refX) ? 0 : refX - half;
      auto endX = std::min(half + refX, size.x - 1);
      auto startY = (half > refY) ? 0 : refY - half;
      auto endY = std::min(half + refY, size.y - 1);
      auto startZ = (half > refZ) ? 0 : refZ - half;
      auto endZ = std::min(half + refZ, size.z - 1);
      float sumLocX = 0;
      float sumLocY = 0;
      float sumLocZ = 0;
      float sumWeight = 0;
      for (auto z = startZ; z <= endZ; ++z) {
        for (auto y = startY; y <= endY; ++y) {
          for (auto x = startX; x <= endX; ++x) {
            auto i = z * size.x * size.y + y * size.x + x;
            auto relVal = ptr[i] / refVal;
            sumWeight += relVal;
            sumLocX += static_cast<float>(x) * relVal;
            sumLocY += static_cast<float>(y) * relVal;
            sumLocZ += static_cast<float>(z) * relVal;
          }
        }
      }
      ptrLoc[0] = sumLocX / sumWeight;
      if (size.GetDimAsNumber() > 1) { ptrLoc[1] = sumLocY / sumWeight; }
      if (size.GetDimAsNumber() > 2) { ptrLoc[2] = sumLocZ / sumWeight; }
    }
    return true;
  }
  // otherwise we don't know what to do, so 'report' it
  for (size_t n = 0; n < size.n * size.GetDimAsNumber(); ++n) {
    locs[n] = std::numeric_limits<float>::quiet_NaN();
  }
  return false;
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
