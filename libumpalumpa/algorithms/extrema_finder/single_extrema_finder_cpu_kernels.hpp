#pragma once

#include <libumpalumpa/data/size.hpp>

namespace umpalumpa {
namespace extrema_finder {

  template<typename T, typename C>
  bool FindSingleExtremaValXDCPU(T *__restrict__ vals,
    T *const __restrict__ data,
    const umpalumpa::data::Size &size, // must be at least 1
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
}// namespace extrema_finder
}// namespace umpalumpa
