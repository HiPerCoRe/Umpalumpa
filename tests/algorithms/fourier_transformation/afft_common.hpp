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

void testFFTInpulseOrigin(AFFT::ResultData &out, AFFT::InputData &in, const Settings &settings) {

  for (size_t n = 0; n < in.data.info.size.n; ++n) {
    // impulse at the origin ...
    *(reinterpret_cast<float*>(in.data.data) + n * in.data.info.paddedSize.single) = 1.f;
  }

  auto ft = FFTCUDA();

  ft.Init(out, in, settings);
  ft.Execute(out, in);
  ft.Synchronize();

  float delta = 0.00001f;
  for (size_t i = 0; i < in.data.info.paddedSize.total; ++i) {
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

  Size size(10, 20, 30, 5);

  PhysicalDescriptor pdIn(size.total * sizeof(float), DataType::kFloat);
  FourierDescriptor ldIn(size, size);

  auto *in = calloc(pdIn.bytes, 1); // FIXME should be pre-allocated before the test, and should be reused in more 
  auto inP = AFFT::InputData(Payload(in, ldIn, pdIn, "Input data"));

  PhysicalDescriptor pdOut(inP.data.info.frequencyDomainSizePadded.total * 2 * sizeof(float), DataType::kFloat);
  FourierDescriptor ldOut(inP.data.info.frequencyDomainSize, inP.data.info.frequencyDomainSizePadded);

  void *out;
  if (settings.IsOutOfPlace()) {
    out = calloc(pdOut.bytes, 1);
  } else {
    out = in;
  }

  auto outP = AFFT::ResultData(Payload(out, ldOut, pdOut, "Result data"));

  testFFTInpulseOrigin(outP, inP, settings);

  free(in);
  if ((void*)in != (void*)out) {
    free(out);
  }
}
