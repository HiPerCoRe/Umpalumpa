#pragma once

#include <libumpalumpa/utils/cuda_compatibility.hpp>

namespace umpalumpa {// must be two namespaces for compatiblity with CUDA
namespace fourier_processing {

  /**
   * Index to frequency.
   * Given an index (position from 0 to n) and a size of the FFT, this function returns the
   * corresponding digital frequency (-1/2 to 1/2).
   */
  CUDA_HD float Idx2Freq(size_t idx, size_t size)
  {
    if (size <= 1) return 0;
    auto toT = [](auto t) { return static_cast<float>(t); };
    // TODO if we know that we're working with hermitial half, we can simplify this
    float tmp = (idx <= (size / 2)) ? toT(idx) : (toT(idx) - toT(size));
    return tmp / toT(size);
  }
}// namespace fourier_processing
}// namespace umpalumpa
