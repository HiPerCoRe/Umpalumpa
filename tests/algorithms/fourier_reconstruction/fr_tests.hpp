#pragma once

using namespace umpalumpa::fourier_reconstruction;

TEST_F(NAME, XYPlane)
{
  // auto locality = Locality::kOutOfPlace;
  auto settings = Settings{};

  auto size = Size(5, 5, 1, 1);

  SetUp(settings, size);

  auto out = AFR::OutputData(*pVolume, *pWeight);
  auto in = AFR::InputData(*pFFT, *pVolume, *pWeight);

  // testFP(out, in, settings);

  auto &alg = GetAlg();
  ASSERT_TRUE(alg.Init(out, in, settings));
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();
  ASSERT_TRUE(true);
}
