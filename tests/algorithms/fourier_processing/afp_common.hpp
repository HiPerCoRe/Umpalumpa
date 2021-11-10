#pragma once

#include <iostream>
#include <gtest/gtest.h>
#include <libumpalumpa/algorithms/fourier_transformation/settings.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>
#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>
#include <complex>

namespace ft = umpalumpa::fourier_transformation;
using namespace umpalumpa::fourier_processing;
using namespace umpalumpa::data;

TEST_F(NAME, ImageCropping)
{
  ft::Locality locality = ft::Locality::kOutOfPlace;
  Settings settings(locality);

  Size inSize(10, 10, 1, 1);
  Size outSize(5, 5, 1, 1);

  SetUpFP(settings, inSize, outSize);

  auto inP =
    AFP::InputData(Payload(*ldIn, *pdIn, "Input data"), Payload(*ldFilter, *pdFilter, "Filter"));
  auto outP = AFP::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testFP(outP, inP, settings);
}

TEST_F(NAME, ImageNoCropping)
{
  ft::Locality locality = ft::Locality::kOutOfPlace;
  Settings settings(locality);

  Size size(10, 10, 1, 1);

  SetUpFP(settings, size, size);

  auto inP =
    AFP::InputData(Payload(*ldIn, *pdIn, "Input data"), Payload(*ldFilter, *pdFilter, "Filter"));
  auto outP = AFP::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testFP(outP, inP, settings);
}

TEST_F(NAME, ImageNormalize)
{
  ft::Locality locality = ft::Locality::kOutOfPlace;
  Settings settings(locality);
  settings.SetNormalize(true);

  Size size(10, 10, 1, 1);

  SetUpFP(settings, size, size);

  auto inP =
    AFP::InputData(Payload(*ldIn, *pdIn, "Input data"), Payload(*ldFilter, *pdFilter, "Filter"));
  auto outP = AFP::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testFP(outP, inP, settings);
}

TEST_F(NAME, ImageCentering)
{
  ft::Locality locality = ft::Locality::kOutOfPlace;
  Settings settings(locality);
  settings.SetCenter(true);

  Size size(12, 12, 1, 1);

  SetUpFP(settings, size, size);

  auto inP =
    AFP::InputData(Payload(*ldIn, *pdIn, "Input data"), Payload(*ldFilter, *pdFilter, "Filter"));
  auto outP = AFP::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testFP(outP, inP, settings);
}

TEST_F(NAME, ImageCropNormalizeCenter)
{
  ft::Locality locality = ft::Locality::kOutOfPlace;
  Settings settings(locality);
  settings.SetNormalize(true);
  settings.SetCenter(true);

  Size inSize(40, 40, 1, 5);
  Size outSize(20, 20, 1, 5);

  SetUpFP(settings, inSize, outSize);

  auto inP =
    AFP::InputData(Payload(*ldIn, *pdIn, "Input data"), Payload(*ldFilter, *pdFilter, "Filter"));
  auto outP = AFP::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testFP(outP, inP, settings);
}

TEST_F(NAME, Filtering)
{
  ft::Locality locality = ft::Locality::kOutOfPlace;
  Settings settings(locality);
  settings.SetApplyFilter(true);

  Size size(12, 12, 1, 1);

  SetUpFP(settings, size, size);

  auto inP =
    AFP::InputData(Payload(*ldIn, *pdIn, "Input data"), Payload(*ldFilter, *pdFilter, "Filter"));
  auto outP = AFP::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testFP(outP, inP, settings);
}
