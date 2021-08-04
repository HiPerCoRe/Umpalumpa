#pragma once

#include <iostream>
#include <gtest/gtest.h>
#include <libumpalumpa/algorithms/fourier_transformation/settings.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>
#include <complex>

using namespace umpalumpa::fourier_transformation;
using namespace umpalumpa::data;

template<typename T> void PrintData(T *data, const Size size)
{
  ASSERT_EQ(size.GetDim(), Dimensionality::k2Dim);
  for (size_t n = 0; n < size.n; ++n) {
    size_t offset = n * size.single;
    for (size_t y = 0; y < size.y; ++y) {
      for (size_t x = 0; x < size.x; ++x) { printf("%+.3f\t", data[offset + y * size.x + x]); }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
}

template <>
void PrintData(std::complex<float> *data, const Size size)
{
  ASSERT_EQ(size.GetDim(), Dimensionality::k2Dim);
  for (size_t n = 0; n < size.n; ++n) {
    size_t offset = n * size.single;
    for (size_t y = 0; y < size.y; ++y) {
      for (size_t x = 0; x < size.x; ++x) {
        auto v = data[offset + y * size.x + x];
        printf("(%+.3f,%+.3f)\t", v.real(), v.imag() );
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
}

void testFFTInpulseOrigin(AFFT::ResultData &out, AFFT::InputData &in, const Settings &settings) {

  for (size_t n = 0; n < in.data.info.size.n; ++n) {
    // impulse at the origin ...
    *(reinterpret_cast<float*>(in.data.data) + n * in.data.info.paddedSize.single) = 1.f;
  }

  PrintData((float*)in.data.data, in.data.info.size);

  auto ft = FFTCUDA();

  ft.Init(out, in, settings);
  ft.Execute(out, in);
  ft.Synchronize();

  PrintData((std::complex<float>*)out.data.data, out.data.info.frequencyDomainSizePadded);

  float delta = 0.00001f;
  for (size_t i = 0; i < out.data.info.frequencyDomainSizePadded.total; ++i) {
    // ... will result in constant real value, and no imag value
    auto *tmp = reinterpret_cast<std::complex<float>*>(out.data.data);
    ASSERT_NEAR(1, tmp[i].real(), delta) << " at " << i;
    ASSERT_NEAR(0, tmp[i].imag(), delta) << " at " << i;
  }

}

TEST_F(NAME, test_1)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 1);

  PhysicalDescriptor pdIn(size.total * sizeof(float), DataType::kFloat);
  FourierDescriptor ldIn(size, size);

  void *in = Allocate(pdIn.bytes);
  memset(in, 0, pdIn.bytes);
  //CudaErrchk(cudaMemset(in, 0, pdIn.bytes))
  //auto *in = calloc(pdIn.bytes, 1); // FIXME should be pre-allocated before the test, and should be reused in more 
  auto inP = AFFT::InputData(Payload(in, ldIn, pdIn, "Input data"));

  FourierDescriptor ldOut(ldIn); // copy, because they describe the same data
  PhysicalDescriptor pdOut(ldOut.frequencyDomainSizePadded.total * 2 * sizeof(float), DataType::kFloat);

  void *out;
  if (settings.IsOutOfPlace()) {
    out = Allocate(pdOut.bytes);
    //CudaErrchk(cudaMemset(out, 0, pdIn.bytes))
    //out = calloc(pdOut.bytes, 1);
  } else {
    out = in;
  }

  auto outP = AFFT::ResultData(Payload(out, ldOut, pdOut, "Result data"));

  testFFTInpulseOrigin(outP, inP, settings);

  Free(in);
  if ((void*)in != (void*)out) {
    Free(out);
  }
}
