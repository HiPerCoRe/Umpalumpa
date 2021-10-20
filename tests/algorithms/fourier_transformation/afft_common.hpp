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

TEST_F(NAME, InpulseOriginForward)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 1);

  SetUpFFT(settings, size, size);

  auto inP = AFFT::InputData(Payload(dataSpatial.get(), *ldSpatial, *pdSpatial, "Input data"));
  auto outP = AFFT::OutputData(Payload(dataFrequency.get(), *ldFrequency, *pdFrequency, "Result data"));

  testFFTInpulseOrigin(outP, inP, settings);
}

TEST_F(NAME, InpulseOriginInverse)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kInverse;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 1);

  SetUpFFT(settings, size, size);

  auto inP = AFFT::InputData(Payload(dataFrequency.get(), *ldFrequency, *pdFrequency, "Result data"));
  auto outP = AFFT::OutputData(Payload(dataSpatial.get(), *ldSpatial, *pdSpatial, "Input data"));

  testIFFTInpulseOrigin(outP, inP, settings);
}

TEST_F(NAME, InpulseShiftedForward)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 1);

  SetUpFFT(settings, size, size);

  auto inP = AFFT::InputData(Payload(dataSpatial.get(), *ldSpatial, *pdSpatial, "Input data"));
  auto outP = AFFT::OutputData(Payload(dataFrequency.get(), *ldFrequency, *pdFrequency, "Result data"));

  testFFTInpulseShifted(outP, inP, settings);
}

TEST_F(NAME, FFTIFFT)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 1);

  SetUpFFT(settings, size, size);

  auto inP = AFFT::InputData(Payload(dataSpatial.get(), *ldSpatial, *pdSpatial, "Input data"));
  auto outP = AFFT::OutputData(Payload(dataFrequency.get(), *ldFrequency, *pdFrequency, "Result data"));

  testFFTIFFT(outP, inP, settings);
}
