#pragma once

using namespace umpalumpa::fourier_transformation;

TEST_F(NAME, ImageCropping)
{
  auto locality = Locality::kOutOfPlace;
  auto settings = Settings(locality);

  auto inSize = Size(10, 10, 1, 1);
  auto outSize = Size(5, 5, 1, 1);

  SetUp(settings, inSize, outSize);

  auto out = AFP::OutputData(*pOut);
  auto in = AFP::InputData(*pIn, *pFilter);
  
  testFP(out, in, settings);
}

TEST_F(NAME, ImageNoCropping)
{
  auto locality = Locality::kOutOfPlace;
  auto settings = Settings(locality);

  Size size(10, 10, 1, 1);

  SetUp(settings, size, size);

  auto out = AFP::OutputData(*pOut);
  auto in = AFP::InputData(*pIn, *pFilter);
  
  testFP(out, in, settings);
}

TEST_F(NAME, ImageNormalize)
{
  auto locality = Locality::kOutOfPlace;
  auto settings = Settings(locality);
  settings.SetNormalize(true);

  Size size(10, 10, 1, 1);

  SetUp(settings, size, size);

  auto out = AFP::OutputData(*pOut);
  auto in = AFP::InputData(*pIn, *pFilter);
  
  testFP(out, in, settings);
}

TEST_F(NAME, ImageCentering)
{
  auto locality = Locality::kOutOfPlace;
  auto settings = Settings(locality);
  settings.SetCenter(true);

  Size size(12, 12, 1, 1);

  SetUp(settings, size, size);

  auto out = AFP::OutputData(*pOut);
  auto in = AFP::InputData(*pIn, *pFilter);
  
  testFP(out, in, settings);
}

TEST_F(NAME, ImageCropNormalizeCenter)
{
  auto locality = Locality::kOutOfPlace;
  auto settings = Settings(locality);
  settings.SetNormalize(true);
  settings.SetCenter(true);

  Size inSize(40, 40, 1, 5);
  Size outSize(20, 20, 1, 5);

  SetUp(settings, inSize, outSize);

  auto out = AFP::OutputData(*pOut);
  auto in = AFP::InputData(*pIn, *pFilter);
  
  testFP(out, in, settings);
}

TEST_F(NAME, Filtering)
{
  auto locality = Locality::kOutOfPlace;
  auto settings = Settings(locality);
  settings.SetApplyFilter(true);

  Size size(12, 12, 1, 1);

  SetUp(settings, size, size);

  auto out = AFP::OutputData(*pOut);
  auto in = AFP::InputData(*pIn, *pFilter);
  
  testFP(out, in, settings);
}
