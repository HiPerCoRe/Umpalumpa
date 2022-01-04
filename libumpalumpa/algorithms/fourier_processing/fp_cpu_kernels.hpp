#pragma once

#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_common_kernels.hpp>

namespace umpalumpa::fourier_processing {

template<bool applyFilter, bool normalize, bool center, bool cropFreq> struct ScaleFFT2DCPU
{
  template<typename T, typename T2>
  static void Execute(const T2 *__restrict__ in,
    T2 *__restrict__ out,
    const umpalumpa::data::Size &inSize,
    const umpalumpa::data::Size &inSpatialSize,
    const umpalumpa::data::Size &outSize,
    const T *__restrict__ filter,
    float normFactor,
    float maxFreqSquare)
  {
    for (size_t n = 0; n < inSize.n; ++n) {
      for (size_t y = 0; y < outSize.y; ++y) {
        for (size_t x = 0; x < outSize.x; ++x) {
          size_t origY = (y <= outSize.y / 2) ? y : (inSize.y - (outSize.y - y));
          size_t iIndex = n * inSize.single + origY * inSize.x + x;
          size_t oIndex = n * outSize.single + y * outSize.x + x;
          T2 freq = { Idx2Freq<T>(x, inSpatialSize.x), Idx2Freq<T>(origY, inSpatialSize.y) };
          out[oIndex] = (cropFreq && (norm(freq) > maxFreqSquare)) ? T2{} : in[iIndex];
          if (applyFilter) { out[oIndex] *= filter[y * outSize.x + x]; }
          if (normalize) { out[oIndex] *= normFactor; }
          if (center) {
            out[oIndex] *=
              static_cast<T>(1 - 2 * ((static_cast<int>(x) + static_cast<int>(y)) & 1));
          }
        }
      }
    }
  }
};

}// namespace umpalumpa::fourier_processing
