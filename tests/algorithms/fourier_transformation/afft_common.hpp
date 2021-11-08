#pragma once

#include <iostream>
#include <gtest/gtest.h>
#include <libumpalumpa/algorithms/fourier_transformation/settings.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>
#include <complex>
#include <thread>

using namespace umpalumpa::fourier_transformation;
using namespace umpalumpa::data;

TEST_F(NAME, InpulseOriginForward)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 1);

  SetUpFFT(settings, size, PaddingDescriptor());

  auto inP = AFFT::InputData(Payload(*ldSpatial, *pdSpatial, "Input data"));
  auto outP = AFFT::OutputData(Payload(*ldFrequency, *pdFrequency, "Result data"));

  testFFTInpulseOrigin(outP, inP, settings);
}

TEST_F(NAME, InpulseOriginInverse)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kInverse;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 1);

  SetUpFFT(settings, size, PaddingDescriptor());

  auto inP = AFFT::InputData(Payload(*ldFrequency, *pdFrequency, "Input data"));
  auto outP = AFFT::OutputData(Payload(*ldSpatial, *pdSpatial, "Result data"));

  testIFFTInpulseOrigin(outP, inP, settings);
}

TEST_F(NAME, InpulseShiftedForward)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 1);

  SetUpFFT(settings, size, PaddingDescriptor());

  auto inP = AFFT::InputData(Payload(*ldSpatial, *pdSpatial, "Input data"));
  auto outP = AFFT::OutputData(Payload(*ldFrequency, *pdFrequency, "Result data"));

  testFFTInpulseShifted(outP, inP, settings);
}

TEST_F(NAME, FFTIFFT)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 15);

  SetUpFFT(settings, size, PaddingDescriptor());

  auto inP = AFFT::InputData(Payload(*ldSpatial, *pdSpatial, "Input data"));
  auto outP = AFFT::OutputData(Payload(*ldFrequency, *pdFrequency, "Result data"));

  testFFTIFFT(outP, inP, settings, 0);
}


TEST_F(NAME, FFTIFFT_Batch)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 50);

  SetUpFFT(settings, size, PaddingDescriptor());

  auto inP = AFFT::InputData(Payload(*ldSpatial, *pdSpatial, "Input data"));
  auto outP = AFFT::OutputData(Payload(*ldFrequency, *pdFrequency, "Result data"));

  testFFTIFFT(outP, inP, settings, 10);
}

TEST_F(NAME, FFTIFFT_MultipleThreads)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction, std::max(std::thread::hardware_concurrency(), 1u));

  Size size(300, 200, 32, 55);

  SetUpFFT(settings, size, PaddingDescriptor());

  auto inP = AFFT::InputData(Payload(*ldSpatial, *pdSpatial, "Input data"));
  auto outP = AFFT::OutputData(Payload(*ldFrequency, *pdFrequency, "Result data"));

  testFFTIFFT(outP, inP, settings);
}

TEST_F(NAME, FFTIFFT_BatchNotDivisible)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 53);

  SetUpFFT(settings, size, PaddingDescriptor());

  auto inP = AFFT::InputData(Payload(*ldSpatial, *pdSpatial, "Input data"));
  auto outP = AFFT::OutputData(Payload(*ldFrequency, *pdFrequency, "Result data"));

  testFFTIFFT(outP, inP, settings, 10);
}