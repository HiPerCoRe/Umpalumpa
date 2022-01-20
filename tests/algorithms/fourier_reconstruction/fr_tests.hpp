#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/fr_common_kernels.hpp>
using namespace umpalumpa::fourier_reconstruction;

void TestTravelSpace5x6XYFast(const TraverseSpace &s)
{
  auto TestPoint = [](const auto &l, auto x, auto y, auto z) {
    ASSERT_FLOAT_EQ(l.x, x);
    ASSERT_FLOAT_EQ(l.y, y);
    ASSERT_FLOAT_EQ(l.z, z);
  };

  ASSERT_EQ(s.minY, 0);
  ASSERT_EQ(s.minX, 3);
  ASSERT_EQ(s.minZ, 3);
  ASSERT_EQ(s.maxY, 6);
  ASSERT_EQ(s.maxX, 6);
  ASSERT_EQ(s.maxZ, 3);

  ASSERT_FLOAT_EQ(s.maxDistanceSqr, 9);
  ASSERT_EQ(s.dir, TraverseSpace::Direction::XY);

  TestPoint(s.unitNormal, 0.f, 0.f, 1.f);
  TestPoint(s.topOrigin, 6.f, 0.f, 3.f);
  TestPoint(s.bottomOrigin, 6.f, 0.f, 3.f);

  ASSERT_FLOAT_EQ(s.weight, 1.f);
}

void TestTravelSpace5x6XYPrecise(const TraverseSpace &s)
{
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

void TestResult(const AFR::OutputData &d, const TraverseSpace &s, float maxDistance)
{
  auto &volume = d.GetVolume();

  auto &volumeSize = volume.info.GetSize();
  auto ExpectZero = [&volumeSize](auto *v, size_t x, size_t y, size_t z) {
    size_t index = z * volumeSize.x * volumeSize.y + y * volumeSize.x + x;
    auto c = reinterpret_cast<std::complex<float> *>(v)[index];
    EXPECT_FLOAT_EQ(c.real(), 0.f) << " at " << x << " " << y << " " << z;
    EXPECT_FLOAT_EQ(c.imag(), 0.f) << " at " << x << " " << y << " " << z;
  };
  auto ExpectNotZero = [&volumeSize](auto *v, size_t x, size_t y, size_t z) {
    size_t index = z * volumeSize.x * volumeSize.y + y * volumeSize.x + x;
    auto c = reinterpret_cast<std::complex<float> *>(v)[index];
    EXPECT_NE(c.real(), 0.f) << " at " << x << " " << y << " " << z;
    EXPECT_NE(c.imag(), 0.f) << " at " << x << " " << y << " " << z;
  };
  Point3D<float> imgPos;
  for (int z = 0; z < volumeSize.z; ++z) {
    imgPos.z = static_cast<float>(z - (static_cast<int>(volumeSize.z) - 1) / 2);
    for (int y = 0; y < volumeSize.y; ++y) {
      imgPos.y = static_cast<float>(y - (static_cast<int>(volumeSize.y) - 1) / 2);
      for (int x = 0; x < volumeSize.x; ++x) {
        // transform current point to center
        imgPos.x = static_cast<float>(x - (static_cast<int>(volumeSize.x) - 1) / 2);
        // iterations that would access pixel with too high frequency should be 0
        if (imgPos.x * imgPos.x + imgPos.y * imgPos.y + imgPos.z * imgPos.z > s.maxDistanceSqr) {
          ExpectZero(volume.GetPtr(), x, y, z);
          continue;
        }
        auto posInVolume = imgPos;// because we just shifted the center
        // rotate around center
        multiply(s.transformInv, imgPos);
        if (imgPos.x < -maxDistance
            || std::abs(s.unitNormal.x * posInVolume.x + s.unitNormal.y * posInVolume.y
                        + s.unitNormal.z * posInVolume.z)
                 > maxDistance) {
          // reading outside of the image OR too far from the plane
          // -> should not affect the volume
          ExpectZero(volume.GetPtr(), x, y, z);
        } else {
          // reading from the image -> should affect the volume
          ExpectNotZero(volume.GetPtr(), x, y, z);
        }
      }
    }
  }
}

TEST_F(NAME, XYPlane)
{
  // auto locality = Locality::kOutOfPlace;
  auto settings = Settings{};
  settings.SetType(Settings::Type::kPrecise);
  settings.SetInterpolation(Settings::Interpolation::kDynamic);

  auto size = Size(5, 6, 1, 1);

  SetUp(settings, size);

  float t[3][3] = {};
  t[0][0] = t[1][1] = t[2][2] = 1.f;
  auto &space = *reinterpret_cast<TraverseSpace *>(pTraverseSpace->GetPtr());
  FillTraverseSpace(t, space, pFFT->info.GetSize(), pVolume->info.GetSize(), settings, 1.f);

  settings.GetType() == Settings::Type::kFast ? TestTravelSpace5x6XYFast(space)
                                              : TestTravelSpace5x6XYPrecise(space);

  auto out = AFR::OutputData(*pVolume, *pWeight);
  auto in = AFR::InputData(*pFFT, *pVolume, *pWeight, *pTraverseSpace);


  FillConstant(reinterpret_cast<float *>(pFFT->GetPtr()), pFFT->info.Elems() * 2, 1.f);

  Print(reinterpret_cast<std::complex<float> *>(pFFT->GetPtr()), pFFT->info.GetSize());


  auto &alg = GetAlg();
  ASSERT_TRUE(alg.Init(out, in, settings));
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  Print(reinterpret_cast<std::complex<float> *>(pVolume->GetPtr()), pVolume->info.GetSize());

  TestResult(out, space, settings.GetType() == Settings::Type::kFast ? 0 : settings.GetBlobRadius());

  ASSERT_TRUE(true);
}
