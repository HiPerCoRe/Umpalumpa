#pragma once

// TODO tests:
//  - correct correlations
//  - centering
//  - center + odd sized image -> not working

TEST_F(NAME, CorrelationOnetoOneIntraBufferNoCenter)
{
  Settings settings(CorrelationType::kOneToN);
  settings.SetCenter(false);

  Size inSize(20, 20, 1, 2);

  SetUp(settings, inSize, inSize, true);

  auto in = ACorrelation::InputData(*pData1, *pData2);
  auto out = ACorrelation::OutputData(*pOut);

  testCorrelationSimple(out, in, settings);
}

TEST_F(NAME, CorrelationOnetoOneInterBufferNoCenter)
{
  Settings settings(CorrelationType::kOneToN);
  settings.SetCenter(false);

  Size inSize(20, 20, 1, 1);

  SetUp(settings, inSize, inSize, false);

  auto in = ACorrelation::InputData(*pData1, *pData2);
  auto out = ACorrelation::OutputData(*pOut);

  testCorrelationSimple(out, in, settings);
}

TEST_F(NAME, CorrelationOnetoOneIntraBuffer)
{
  Settings settings(CorrelationType::kOneToN);

  Size inSize(20, 20, 1, 2);

  SetUp(settings, inSize, inSize, true);

  auto in = ACorrelation::InputData(*pData1, *pData2);
  auto out = ACorrelation::OutputData(*pOut);

  testCorrelationSimple(out, in, settings);
}

TEST_F(NAME, CorrelationOnetoOneInterBuffer)
{
  Settings settings(CorrelationType::kOneToN);

  Size inSize(20, 20, 1, 1);

  SetUp(settings, inSize, inSize, false);

  auto in = ACorrelation::InputData(*pData1, *pData2);
  auto out = ACorrelation::OutputData(*pOut);

  testCorrelationSimple(out, in, settings);
}

TEST_F(NAME, CorrelationMtoNIntraBuffer)
{
  Settings settings(CorrelationType::kMToN);

  Size inSize(20, 20, 1, 5);

  SetUp(settings, inSize, inSize, true);

  auto in = ACorrelation::InputData(*pData1, *pData2);
  auto out = ACorrelation::OutputData(*pOut);

  testCorrelationSimple(out, in, settings);
}

TEST_F(NAME, CorrelationMtoNInterBuffer)
{
  Settings settings(CorrelationType::kMToN);

  Size inSize(20, 20, 1, 3);

  SetUp(settings, inSize, inSize.CopyFor(2), false);

  auto in = ACorrelation::InputData(*pData1, *pData2);
  auto out = ACorrelation::OutputData(*pOut);

  testCorrelationSimple(out, in, settings);
}

TEST_F(NAME, CorrelationMtoNIntraBufferRandomData)
{
  Settings settings(CorrelationType::kMToN);

  Size inSize(20, 20, 1, 5);

  SetUp(settings, inSize, inSize, true);

  auto in = ACorrelation::InputData(*pData1, *pData2);
  auto out = ACorrelation::OutputData(*pOut);

  testCorrelationRandomData(out, in, settings);
}

TEST_F(NAME, CorrelationMtoNInterBufferRandomData)
{
  Settings settings(CorrelationType::kMToN);

  Size inSize(20, 20, 1, 3);

  SetUp(settings, inSize, inSize.CopyFor(2), false);

  auto in = ACorrelation::InputData(*pData1, *pData2);
  auto out = ACorrelation::OutputData(*pOut);

  testCorrelationRandomData(out, in, settings);
}
