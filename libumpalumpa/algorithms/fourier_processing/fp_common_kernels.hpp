#pragma once

#include <libumpalumpa/system_includes/cuda_runtime.hpp>

namespace umpalumpa::fourier_processing {

/**
 * Index to frequency.
 * Given an index (position from 0 to n) and a size of the FFT, this function returns the
 * corresponding digital frequency (-1/2 to 1/2).
 */
__host__ __device__ template<typename T> T Idx2Freq(size_t idx, size_t size)
{
  static_assert(std::is_floating_point<T>::value);
  if (size <= 1) return 0;
  auto toT = [](auto t) {
    return static_cast<T>(t);
  };
  T tmp = (idx <= (size / 2)) ? toT(idx) : (toT(idx) - toT(size));
  return tmp / static_cast<T>(size);
}

}// namespace umpalumpa::fourier_processing
