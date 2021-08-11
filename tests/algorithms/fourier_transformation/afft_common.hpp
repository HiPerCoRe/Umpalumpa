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

  FourierDescriptor ldOut(size, size, FourierDescriptor::FourierSpaceDescriptor()); // copy, because they describe the same data
  PhysicalDescriptor pdOut(ldOut.GetPaddedSize().total * 2 * sizeof(float), DataType::kFloat);

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
