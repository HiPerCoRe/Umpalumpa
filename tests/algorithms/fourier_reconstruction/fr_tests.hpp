#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/fr_common_kernels.hpp>
using namespace umpalumpa::fourier_reconstruction;

// TODO add test for constants

TEST_F(NAME, XYPlanePreciseDynamic)
{
  auto settings = Settings{};
  settings.SetType(Settings::Type::kPrecise);
  settings.SetInterpolation(Settings::Interpolation::kDynamic);
  TestXYPlane5x6(settings);
}

TEST_F(NAME, XYPlaneFastDynamic)
{
  auto settings = Settings{};
  settings.SetType(Settings::Type::kFast);
  settings.SetInterpolation(Settings::Interpolation::kDynamic);
  TestXYPlane5x6(settings);
}

TEST_F(NAME, XYPlanePreciseLookup)
{
  auto settings = Settings{};
  settings.SetType(Settings::Type::kPrecise);
  settings.SetInterpolation(Settings::Interpolation::kLookup);
  TestXYPlane5x6(settings);
}

TEST_F(NAME, XYPlaneFastLookup)
{
  auto settings = Settings{};
  settings.SetType(Settings::Type::kFast);
  settings.SetInterpolation(Settings::Interpolation::kLookup);
  TestXYPlane5x6(settings);
}

TEST_F(NAME, YZPlanePreciseDynamic)
{
  auto settings = Settings{};
  settings.SetType(Settings::Type::kPrecise);
  settings.SetInterpolation(Settings::Interpolation::kDynamic);
  TestYZPlane5x6(settings);
}

TEST_F(NAME, YZPlaneFastDynamic)
{
  auto settings = Settings{};
  settings.SetType(Settings::Type::kFast);
  settings.SetInterpolation(Settings::Interpolation::kDynamic);
  TestYZPlane5x6(settings);
}

TEST_F(NAME, YZPlanePreciseLookup)
{
  auto settings = Settings{};
  settings.SetType(Settings::Type::kPrecise);
  settings.SetInterpolation(Settings::Interpolation::kLookup);
  TestYZPlane5x6(settings);
}

TEST_F(NAME, YZPlaneFastLookup)
{
  auto settings = Settings{};
  settings.SetType(Settings::Type::kFast);
  settings.SetInterpolation(Settings::Interpolation::kLookup);
  TestYZPlane5x6(settings);
}
