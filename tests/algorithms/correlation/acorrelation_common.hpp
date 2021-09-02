#pragma once

#include <gtest/gtest.h>
#include <libumpalumpa/algorithms/correlation/acorrelation.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>

using namespace umpalumpa::correlation;
using namespace umpalumpa::data;

//TODO tests:
//  - correct correlations
//  - centering
//  - center + odd sized image -> not working

TEST_F(NAME, CorrelationOnetoOneIntraBufferNoCenter)
{
  Settings settings(CorrelationType::kOneToN);
  settings.SetCenter(false);

  Size inSize(10, 10, 1, 2);

  SetUpCorrelation(settings, inSize);

  auto inP = ACorrelation::InputData(Payload(inData1.get(), *ldIn1, *pdIn1, "Input data 1"), Payload(inData2.get(), *ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(outData.get(), *ldOut, *pdOut, "Output data"));

  testCorrelation(outP, inP, settings);
}

TEST_F(NAME, CorrelationOnetoOneInterBufferNoCenter)
{
  Settings settings(CorrelationType::kOneToN);
  settings.SetCenter(false);

  Size inSize(10, 10, 1, 1);

  SetUpCorrelation(settings, inSize, 1);

  auto inP = ACorrelation::InputData(Payload(inData1.get(), *ldIn1, *pdIn1, "Input data 1"), Payload(inData2.get(), *ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(outData.get(), *ldOut, *pdOut, "Output data"));

  testCorrelation(outP, inP, settings);
}

TEST_F(NAME, CorrelationOnetoOneIntraBuffer)
{
  Settings settings(CorrelationType::kOneToN);

  Size inSize(10, 10, 1, 2);

  SetUpCorrelation(settings, inSize);

  auto inP = ACorrelation::InputData(Payload(inData1.get(), *ldIn1, *pdIn1, "Input data 1"), Payload(inData2.get(), *ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(outData.get(), *ldOut, *pdOut, "Output data"));

  testCorrelation(outP, inP, settings);
}

TEST_F(NAME, CorrelationOnetoOneInterBuffer)
{
  Settings settings(CorrelationType::kOneToN);

  Size inSize(10, 10, 1, 1);

  SetUpCorrelation(settings, inSize, 1);

  auto inP = ACorrelation::InputData(Payload(inData1.get(), *ldIn1, *pdIn1, "Input data 1"), Payload(inData2.get(), *ldIn2, *pdIn2, "Input data 2"));
  auto outP = ACorrelation::OutputData(Payload(outData.get(), *ldOut, *pdOut, "Output data"));

  testCorrelation(outP, inP, settings);
}

