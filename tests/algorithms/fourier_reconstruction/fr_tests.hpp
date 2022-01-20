#pragma once

using namespace umpalumpa::fourier_reconstruction;

void TestTravelSpace(const TraverseSpace &s) {
  auto TestPoint = [](const auto &l, auto x, auto y, auto z) {
    ASSERT_FLOAT_EQ(l.x, x);
    ASSERT_FLOAT_EQ(l.y, y);
    ASSERT_FLOAT_EQ(l.z, z);
  };
  
  ASSERT_EQ(s.minY, 0);
  ASSERT_EQ(s.minX, 1);
  ASSERT_EQ(s.minZ, 1);
  ASSERT_EQ(s.maxY, 6);
  ASSERT_EQ(s.maxX, 6);
  ASSERT_EQ(s.maxZ, 5);

  ASSERT_FLOAT_EQ(s.maxDistanceSqr, 24.0100002);
  ASSERT_EQ(s.dir, TraverseSpace::Direction::XY);

  TestPoint(s.unitNormal, 0.f, 0.f, 1.f);
  TestPoint(s.topOrigin, 7.9f, -1.9f, 1.1f);
  TestPoint(s.bottomOrigin, 7.9f, -1.9f, 4.9f);

  ASSERT_FLOAT_EQ(s.weight, 1.f);
}

TEST_F(NAME, XYPlane)
{
  // auto locality = Locality::kOutOfPlace;
  auto settings = Settings{};
  settings.SetInterpolation(Settings::Interpolation::kDynamic);
  settings.SetType(Settings::Type::kPrecise);

  auto size = Size(5, 6, 1, 1);

  SetUp(settings, size);

  float t[3][3] = {};
  t[0][0] = t[1][1] = t[2][2] = 1.f;
  auto &space = *reinterpret_cast<TraverseSpace *>(pTraverseSpace->GetPtr());
  FillTraverseSpace(t, space, pFFT->info.GetSize(), pVolume->info.GetSize(), settings, 1.f);

  TestTravelSpace(space);

  auto out = AFR::OutputData(*pVolume, *pWeight);
  auto in = AFR::InputData(*pFFT, *pVolume, *pWeight, *pTraverseSpace);


  FillIncreasing(reinterpret_cast<float*>(pFFT->GetPtr()), pFFT->info.Elems() * 2, 0.f);

  Print(reinterpret_cast<std::complex<float>*>(pFFT->GetPtr()), pFFT->info.GetSize());


  auto &alg = GetAlg();
  ASSERT_TRUE(alg.Init(out, in, settings));
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  Print(reinterpret_cast<std::complex<float>*>(pVolume->GetPtr()), pVolume->info.GetSize());

  ASSERT_TRUE(true);
}
