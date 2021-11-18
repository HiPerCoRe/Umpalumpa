#pragma once

#include <libumpalumpa/data/size.hpp>

namespace umpalumpa::extrema_finder {

template<typename T, typename C>
bool FindSingleExtremaValXDCPU(T *__restrict__ vals,
  T *const __restrict__ data,
  const umpalumpa::data::Size &size,// must be at least 1
  const C &comp)
{
  // all checks are expected to be done by caller
  for (size_t n = 0; n < size.n; ++n) {
    const size_t offset = n * size.single;
    auto extrema = data[offset];
    for (size_t i = 1; i < size.single; ++i) {
      auto &v = data[offset + i];
      if (comp(v, extrema)) { extrema = v; }
    }
    vals[n] = extrema;
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
