#pragma once
#include <thread>

TEST_F(NAME, InpulseOriginForward)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 1);

  SetUp(settings, size);

  auto inP = AFFT::InputData(*pSpatial);
  auto outP = AFFT::OutputData(*pFrequency);

  testFFTInpulseOrigin(outP, inP, settings);
}

TEST_F(NAME, InpulseOriginInverse)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kInverse;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 1);

  SetUp(settings, size);

  auto inP = AFFT::InputData(*pFrequency);
  auto outP = AFFT::OutputData(*pSpatial);

  testIFFTInpulseOrigin(outP, inP, settings);
}

TEST_F(NAME, InpulseShiftedForward)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 1);

  SetUp(settings, size);

  auto inP = AFFT::InputData(*pSpatial);
  auto outP = AFFT::OutputData(*pFrequency);

  testFFTInpulseShifted(outP, inP, settings);
}

TEST_F(NAME, FFTIFFT)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 15);

  SetUp(settings, size);

  auto inP = AFFT::InputData(*pSpatial);
  auto outP = AFFT::OutputData(*pFrequency);

  testFFTIFFT(outP, inP, settings, 0);
}


TEST_F(NAME, FFTIFFT_Batch)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 50);

  SetUp(settings, size);

  auto inP = AFFT::InputData(*pSpatial);
  auto outP = AFFT::OutputData(*pFrequency);

  testFFTIFFT(outP, inP, settings, 10);
}

TEST_F(NAME, FFTIFFT_MultipleThreads)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction, std::max(std::thread::hardware_concurrency(), 1u));

  Size size(300, 200, 32, 55);

  SetUp(settings, size);

  auto inP = AFFT::InputData(*pSpatial);
  auto outP = AFFT::OutputData(*pFrequency);

  testFFTIFFT(outP, inP, settings);
}

TEST_F(NAME, FFTIFFT_BatchNotDivisible)
{
  Locality locality = Locality::kOutOfPlace;
  Direction direction = Direction::kForward;
  Settings settings(locality, direction);

  Size size(5, 5, 1, 53);

  SetUp(settings, size);

  auto inP = AFFT::InputData(*pSpatial);
  auto outP = AFFT::OutputData(*pFrequency);

  testFFTIFFT(outP, inP, settings, 10);
}