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

template<typename T, unsigned WINDOW>
bool FindSingleExtremaSubPixel1D(T *__restrict__,
  float *__restrict__ locs,
  T *const __restrict__ data,
  const umpalumpa::data::Size &size)
{
  assert(WINDOW % 2 == 1);
  auto half = (WINDOW - 1) / 2;
  // auto tmp = std::make_unique<T>(new LocWeight[WINDOW]);
  for (size_t n = 0; n < size.n; ++n) {
    auto *ptr = data + n * size.single;
    auto refPos = static_cast<size_t>(locs[n]);
    auto refVal = ptr[refPos];
    // careful with unsigned operations
    auto startX = (half > refPos) ? 0 : refPos - half;
    auto endX = std::min(half + refPos, size.x - 1);
    float sumLoc = 0;
    float sumWeight = 0;
    for (auto i = startX; i <= endX; ++i) {
      auto relVal = ptr[i] / refVal;
      sumWeight += relVal;
      auto locWeight = static_cast<float>(i) * relVal;
      sumLoc += locWeight;
    }
    locs[n] = sumLoc / sumWeight;
  }
  return true;
}

template<typename T, unsigned WINDOW>
bool FindSingleExtremaSubPixel2D(T *__restrict__,
  float *__restrict__ locs,
  T *const __restrict__ data,
  const umpalumpa::data::Size &size)
{
  assert(WINDOW % 2 == 1);
  auto half = (WINDOW - 1) / 2;
  // auto tmp = std::make_unique<T>(new LocWeight[WINDOW]);
  for (size_t n = 0; n < size.n; ++n) {
    auto *ptr = data + n * size.single;
    auto refX = static_cast<size_t>(locs[2 * n]);
    auto refY = static_cast<size_t>(locs[2 * n + 1]);
    auto refIndex = refY * size.x + refX;
    auto refVal = ptr[refIndex];
    // careful with unsigned operations
    auto startX = (half > refX) ? 0 : refX - half;
    auto endX = std::min(half + refX, size.x - 1);
    auto startY = (half > refY) ? 0 : refY - half;
    auto endY = std::min(half + refY, size.y - 1);
    float sumLocX = 0;
    float sumLocY = 0;
    float sumWeight = 0;
    for (auto y = startY; y <= endY; ++y) {
      for (auto x = startX; x <= endX; ++x) {
        auto i = y * size.x + x;
        auto relVal = ptr[i] / refVal;
        sumWeight += relVal;
        sumLocX += static_cast<float>(x) * relVal;
        sumLocY += static_cast<float>(y) * relVal;
      }
    }
    locs[2 * n] = sumLocX / sumWeight;
    locs[2 * n + 1] = sumLocY / sumWeight;
  }
  return true;
}

/**
 * Find sub-pixel location or value of the extrema.
 * Data has to contain at least one (1) value.
 * Returned location / value is calculated by relative weigting in the given
 * window using the value contribution. Should the window reach behind the boundaries, those
 *values will be ignored. Only odd sizes of the window are valid.
 *
 * All checks are expected to be done by caller
 **/
template<typename T, unsigned WINDOW>
bool FindSingleExtremaSubPixel(T *__restrict__ vals,
  float *__restrict__ locs,
  T *const __restrict__ data,
  const umpalumpa::data::Size &size)
{
  switch (size.GetDim()) {
  case umpalumpa::data::Dimensionality::k1Dim:
    return FindSingleExtremaSubPixel1D<T, WINDOW>(vals, locs, data, size);
  case umpalumpa::data::Dimensionality::k2Dim:
    return FindSingleExtremaSubPixel2D<T, WINDOW>(vals, locs, data, size);
  default:
    for (size_t n = 0; n < size.n * size.GetDimAsNumber(); ++n) {
      locs[n] = std::numeric_limits<float>::quiet_NaN();
    }
    return false;
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
