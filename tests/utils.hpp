#pragma once

#include <fcntl.h>
#include <random>
#include <complex>
#include <execution>
#include <libumpalumpa/data/size.hpp>

namespace umpalumpa::test {
template<typename T> void FillRandom(T *dst, size_t bytes)
{
  int fd = open("/dev/urandom", O_RDONLY);
  read(fd, dst, bytes);
  close(fd);
}

template<typename T> void FillNormalDist(T *data, size_t elems, T mean = 0, T stddev = 1)
{
  auto mt = std::mt19937(42);
  auto dist = std::normal_distribution<T>(mean, stddev);
  std::for_each(
    std::execution::par_unseq, data, data + elems, [&dist, &mt](auto &&e) { e = dist(mt); });
}

template<typename T> void Print(std::complex<T> *data, const data::Size size)
{
  ASSERT_EQ(size.GetDim(), data::Dimensionality::k2Dim);
  for (size_t n = 0; n < size.n; ++n) {
    size_t offset = n * size.single;
    for (size_t y = 0; y < size.y; ++y) {
      for (size_t x = 0; x < size.x; ++x) {
        auto v = data[offset + y * size.x + x];
        printf("(%+.3f,%+.3f)\t", v.real(), v.imag());
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
}

template<typename T> void PrintData(T *data, const data::Size size)
{
  ASSERT_EQ(size.GetDim(), data::Dimensionality::k2Dim);
  for (size_t n = 0; n < size.n; ++n) {
    size_t offset = n * size.single;
    for (size_t y = 0; y < size.y; ++y) {
      for (size_t x = 0; x < size.x; ++x) { printf("%+.3f\t", data[offset + y * size.x + x]); }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
}
}// namespace umpalumpa::test
