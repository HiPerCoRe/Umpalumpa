#pragma once

#include <gtest/gtest.h>
#include <libumpalumpa/algorithms/correlation/acorrelation.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>

using namespace umpalumpa::correlation;
using namespace umpalumpa::data;

// TODO tests:
//  - correct correlations
//  - centering
//  - center + odd sized image -> not working

TEST_F(NAME, CorrelationOnetoOneIntraBufferNoCenter)
{
  Settings settings(CorrelationType::kOneToN);
  settings.SetCenter(false);

  Size inSize(20, 20, 1, 2);

  SetUpCorrelation(settings, inSize);

  auto inP = ACorrelation::InputData(
    Payload(*ldIn1, *pdIn1, "Input data 1"), Payload(*ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testCorrelationSimple(outP, inP, settings);
}

TEST_F(NAME, CorrelationOnetoOneInterBufferNoCenter)
{
  Settings settings(CorrelationType::kOneToN);
  settings.SetCenter(false);

  Size inSize(20, 20, 1, 1);

  SetUpCorrelation(settings, inSize, 1);

  auto inP = ACorrelation::InputData(
    Payload(*ldIn1, *pdIn1, "Input data 1"), Payload(*ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testCorrelationSimple(outP, inP, settings);
}

TEST_F(NAME, CorrelationOnetoOneIntraBuffer)
{
  Settings settings(CorrelationType::kOneToN);

  Size inSize(20, 20, 1, 2);

  SetUpCorrelation(settings, inSize);

  auto inP = ACorrelation::InputData(
    Payload(*ldIn1, *pdIn1, "Input data 1"), Payload(*ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testCorrelationSimple(outP, inP, settings);
}

TEST_F(NAME, CorrelationOnetoOneInterBuffer)
{
  Settings settings(CorrelationType::kOneToN);

  Size inSize(20, 20, 1, 1);

  SetUpCorrelation(settings, inSize, 1);

  auto inP = ACorrelation::InputData(
    Payload(*ldIn1, *pdIn1, "Input data 1"), Payload(*ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testCorrelationSimple(outP, inP, settings);
}

TEST_F(NAME, CorrelationMtoNIntraBuffer)
{
  Settings settings(CorrelationType::kMToN);

  Size inSize(20, 20, 1, 5);

  SetUpCorrelation(settings, inSize);

  auto inP = ACorrelation::InputData(
    Payload(*ldIn1, *pdIn1, "Input data 1"), Payload(*ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testCorrelationSimple(outP, inP, settings);
}

TEST_F(NAME, CorrelationMtoNInterBuffer)
{
  Settings settings(CorrelationType::kMToN);

  Size inSize(20, 20, 1, 3);

  SetUpCorrelation(settings, inSize, 2);

  auto inP = ACorrelation::InputData(
    Payload(*ldIn1, *pdIn1, "Input data 1"), Payload(*ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testCorrelationSimple(outP, inP, settings);
}

TEST_F(NAME, CorrelationMtoNIntraBufferRandomData)
{
  Settings settings(CorrelationType::kMToN);

  Size inSize(20, 20, 1, 5);

  SetUpCorrelation(settings, inSize);

  auto inP = ACorrelation::InputData(
    Payload(*ldIn1, *pdIn1, "Input data 1"), Payload(*ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testCorrelationRandomData(outP, inP, settings);
}

TEST_F(NAME, CorrelationMtoNInterBufferRandomData)
{
  Settings settings(CorrelationType::kMToN);

  Size inSize(20, 20, 1, 3);

  SetUpCorrelation(settings, inSize, 2);

  auto inP = ACorrelation::InputData(
    Payload(*ldIn1, *pdIn1, "Input data 1"), Payload(*ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(*ldOut, *pdOut, "Output data"));

  testCorrelationRandomData(outP, inP, settings);
}
