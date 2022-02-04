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
  std::ignore = read(fd, dst, bytes);
  close(fd);
}

template<typename T> void FillIncreasing(T *dst, size_t elems, T first)
{
  std::iota(dst, dst + elems, first);
}

template<typename T> void FillConstant(T *dst, size_t elems, T c)
{
  std::fill(dst, dst + elems, c);
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
  std::cout << "Printing data of size " << size.x << " x " << size.y << " x " << size.z << " ("
            << size.n << ")\n";
  for (size_t n = 0; n < size.n; ++n) {
    size_t offset = n * size.single;
    for (size_t z = 0; z < size.z; ++z) {
      for (size_t y = 0; y < size.y; ++y) {
        for (size_t x = 0; x < size.x; ++x) {
          auto v = data[offset + z * (size.x * size.y) + y * size.x + x];
          printf("(% 7.3f,% 7.3f)\t", v.real(), v.imag());
        }
        std::cout << "\n";
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
    for (size_t z = 0; z < size.z; ++z) {
      for (size_t y = 0; y < size.y; ++y) {
        for (size_t x = 0; x < size.x; ++x) {
          printf("%+.3f\t", data[offset + z * size.x * size.y + y * size.x + x]);
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
}
}// namespace umpalumpa::test
