#pragma once

#include <libumpalumpa/data/size.hpp>
#include <complex>

template<typename T, bool center, bool isWithin>
void correlate2D(std::complex<T> *__restrict__ correlations,
  const std::complex<T> *__restrict__ in1,
  umpalumpa::data::Size in1Size,
  const std::complex<T> *__restrict__ in2,
  size_t in2N)
{
  size_t counter = 0;
  // for each image in the first buffer
  for (size_t i = 0; i < in1Size.n; i++) {
    size_t tmpOffset = i * in1Size.single;
    // for each image in the second buffer
    for (size_t j = isWithin ? i + 1 : 0; j < in2N; j++) {
      // for each pixel
      for (size_t y = 0; y < in1Size.y; ++y) {
        for (size_t x = 0; x < in1Size.x; ++x) {
          // center FFT, input must be even
          long centerCoef = 1l - 2l * (static_cast<long>(x + y) & 1l);
          size_t pixelIndex = y * in1Size.x + x;
          auto tmp = in1[tmpOffset + pixelIndex];
          size_t tmp2Offset = j * in1Size.single;
          auto tmp2 = in2[tmp2Offset + pixelIndex];
          std::complex<T> res = { (tmp.real() * tmp2.real()) + (tmp.imag() * tmp2.imag()),
            (tmp.imag() * tmp2.real()) - (tmp.real() * tmp2.imag()) };
          if (center) { res *= static_cast<T>(centerCoef); }
          correlations[counter * in1Size.single + pixelIndex] = res;
        }
      }
      counter++;
    }
  }
}
