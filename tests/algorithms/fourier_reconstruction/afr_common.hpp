#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/afr.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space_generator.hpp>
#include <tests/algorithms/common.hpp>
#include <tests/utils.hpp>

using namespace umpalumpa::fourier_reconstruction;
using namespace umpalumpa::data;
using namespace umpalumpa::test;

/**
 * Class responsible for testing.
 * Specific implementation of the algorithms should inherit from it.
 **/
class FR_Tests : public TestAlg<AFR>
{
protected:
  auto CreatePayloadFFT(const Settings &settings, const Size &size)
  {
    auto fd = FourierDescriptor::FourierSpaceDescriptor{};
    auto ld = FourierDescriptor(size, PaddingDescriptor(), fd);
    auto type = DataType::Get<std::complex<float>>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Input projecttion data in FD");
  }

  auto CreatePayloadVolume(const Settings &settings, const Size &size)
  {
    auto fd = FourierDescriptor::FourierSpaceDescriptor{};
    fd.hasSymetry = true;
    auto ld = FourierDescriptor(size.CopyFor(1), PaddingDescriptor(), fd);
    auto type = DataType::Get<std::complex<float>>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Volume in FD");
  }

  auto CreatePayloadWeights(const Settings &settings, const Size &size)
  {
    auto ld = LogicalDescriptor(size);
    auto type = DataType::Get<float>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Weights");
  }

  void FillTraverseSpace(const float transform[3][3],
    TraverseSpace &space,
    const Size &transformationSize,
    const Size &volumeSize,
    const Settings &s,
    float weight)
  {
    return computeTraverseSpace(
      transformationSize.y
        / 2,// FIXME this should be probably .x, but Xmipp implementation has it like this
      transformationSize.y,
      transform,
      space,
      volumeSize.x - 1,
      volumeSize.y - 1,
      s.GetType() == Settings::Type::kFast,
      s.GetBlobRadius(),
      weight);
  }

  auto CreatePayloadTraverseSpace(const Settings &settings)
  {
    // TODO pass number of spaces needed?
    auto ld = LogicalDescriptor(Size(1, 1, 1, 1));
    auto type = DataType::Get<TraverseSpace>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Traverse space");
  }

  auto CreatePayloadBlobTable(const Settings &settings)
  {
    auto count = settings.GetInterpolation() == Settings::Interpolation::kLookup ? 10000 : 0;
    auto ld = LogicalDescriptor(Size(count, 1, 1, 1));
    auto type = DataType::Get<float>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "BlobTable");
  }

  void SetUp(const Settings &settings, const Size &projectionSize)
  {
    pFFT = std::make_unique<Payload<FourierDescriptor>>(CreatePayloadFFT(settings, projectionSize));
    Register(pFFT->dataInfo);

    // we need uniform cube in the fourier domain
    auto volumeSize = Size(projectionSize.y + 1, projectionSize.y + 1, projectionSize.y + 1, 1);
    pVolume =
      std::make_unique<Payload<FourierDescriptor>>(CreatePayloadVolume(settings, volumeSize));
    Register(pVolume->dataInfo);

    pWeight =
      std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadWeights(settings, volumeSize));
    Register(pWeight->dataInfo);

    pTraverseSpace =
      std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadTraverseSpace(settings));
    Register(pTraverseSpace->dataInfo);

    pBlobTable = std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadBlobTable(settings));
    Register(pBlobTable->dataInfo);
  }

  /**
   * Called at the end of each test fixture
   **/
  void TearDown() override
  {
    auto Clear = [this](auto &p) {
      Unregister(p->dataInfo);
      Remove(p->dataInfo);
    };

    Clear(pFFT);
    Clear(pVolume);
    Clear(pWeight);
    Clear(pTraverseSpace);
    Clear(pBlobTable);
  }
  void TestPoint(const Point3D<float> &l, float x, float y, float z)
  {
    ASSERT_FLOAT_EQ(l.x, x);
    ASSERT_FLOAT_EQ(l.y, y);
    ASSERT_FLOAT_EQ(l.z, z);
  };

  void TestTravelSpace5x6XYFast(const TraverseSpace &s)
  {
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

  void TestTravelSpace5x6YZFast(const TraverseSpace &s)
  {
    ASSERT_EQ(s.minY, 0);
    ASSERT_EQ(s.minX, 3);
    ASSERT_EQ(s.minZ, 0);
    ASSERT_EQ(s.maxY, 6);
    ASSERT_EQ(s.maxX, 3);
    ASSERT_EQ(s.maxZ, 3);

    ASSERT_FLOAT_EQ(s.maxDistanceSqr, 9);
    ASSERT_EQ(s.dir, TraverseSpace::Direction::YZ);

    TestPoint(s.unitNormal, 1.f, 0.f, 0.f);
    TestPoint(s.topOrigin, 3.f, 0.f, 0.f);
    TestPoint(s.bottomOrigin, 3.f, 0.f, 0.f);

    ASSERT_FLOAT_EQ(s.weight, 1.f);
  }

  void TestTravelSpace5x6YZPrecise(const TraverseSpace &s)
  {
    ASSERT_EQ(s.minY, 0);
    ASSERT_EQ(s.minX, 1);
    ASSERT_EQ(s.minZ, 0);
    ASSERT_EQ(s.maxY, 6);
    ASSERT_EQ(s.maxX, 5);
    ASSERT_EQ(s.maxZ, 5);

    ASSERT_FLOAT_EQ(s.maxDistanceSqr, 24.0100002);
    ASSERT_EQ(s.dir, TraverseSpace::Direction::YZ);

    TestPoint(s.unitNormal, 1.f, 0.f, 0.f);
    TestPoint(s.topOrigin, 1.1f, -1.9f, -1.9f);
    TestPoint(s.bottomOrigin, 4.9f, -1.9f, -1.9f);

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
    Point3D<float> posInVolume;
    for (int z = 0; z < volumeSize.z; ++z) {
      posInVolume.z = static_cast<float>(z - (static_cast<int>(volumeSize.z) - 1) / 2);
      for (int y = 0; y < volumeSize.y; ++y) {
        posInVolume.y = static_cast<float>(y - (static_cast<int>(volumeSize.y) - 1) / 2);
        for (int x = 0; x < volumeSize.x; ++x) {
          // transform current point to center
          posInVolume.x = static_cast<float>(x - (static_cast<int>(volumeSize.x) - 1) / 2);
          // iterations that would access pixel with too high frequency should be 0
          if (posInVolume.x * posInVolume.x + posInVolume.y * posInVolume.y + posInVolume.z * posInVolume.z > s.maxDistanceSqr) {
            ExpectZero(volume.GetPtr(), x, y, z);
            continue;
          }
          auto imgPos = posInVolume;// because we just shifted the center
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

  void TestXYPlane5x6(const Settings &settings)
  {
    auto size = Size(5, 6, 1, 1);

    SetUp(settings, size);

    float t[3][3] = {};
    t[0][0] = t[1][1] = t[2][2] = 1.f;
    auto &space = *reinterpret_cast<TraverseSpace *>(pTraverseSpace->GetPtr());
    FillTraverseSpace(t, space, pFFT->info.GetSize(), pVolume->info.GetSize(), settings, 1.f);

    settings.GetType() == Settings::Type::kFast ? TestTravelSpace5x6XYFast(space)
                                                : TestTravelSpace5x6XYPrecise(space);

    auto out = AFR::OutputData(*pVolume, *pWeight);
    auto in = AFR::InputData(*pFFT, *pVolume, *pWeight, *pTraverseSpace, *pBlobTable);

    if (settings.GetInterpolation() == Settings::Interpolation::kLookup) {
      AFR::FillBlobTable(in, settings);
    }

    FillConstant(reinterpret_cast<float *>(pFFT->GetPtr()), pFFT->info.Elems() * 2, 1.f);

    // Print(reinterpret_cast<std::complex<float> *>(pFFT->GetPtr()), pFFT->info.GetSize());

    auto &alg = GetAlg();
    ASSERT_TRUE(alg.Init(out, in, settings));
    ASSERT_TRUE(alg.Execute(out, in));
    // wait till the work is done
    alg.Synchronize();

    // Print(reinterpret_cast<std::complex<float> *>(pVolume->GetPtr()), pVolume->info.GetSize());

    TestResult(
      out, space, settings.GetType() == Settings::Type::kFast ? 0 : settings.GetBlobRadius());
  }

  void TestYZPlane5x6(const Settings &settings)
  {
    auto size = Size(5, 6, 1, 1);

    SetUp(settings, size);

    float t[3][3] = {};
    t[0][2] = t[1][1] = 1.f;
    t[2][0] = -1.f;
    auto &space = *reinterpret_cast<TraverseSpace *>(pTraverseSpace->GetPtr());
    FillTraverseSpace(t, space, pFFT->info.GetSize(), pVolume->info.GetSize(), settings, 1.f);

    settings.GetType() == Settings::Type::kFast ? TestTravelSpace5x6YZFast(space)
                                                : TestTravelSpace5x6YZPrecise(space);

    auto out = AFR::OutputData(*pVolume, *pWeight);
    auto in = AFR::InputData(*pFFT, *pVolume, *pWeight, *pTraverseSpace, *pBlobTable);

    if (settings.GetInterpolation() == Settings::Interpolation::kLookup) {
      AFR::FillBlobTable(in, settings);
    }

    FillIncreasing(reinterpret_cast<float *>(pFFT->GetPtr()), pFFT->info.Elems() * 2, 1.f);

    // Print(reinterpret_cast<std::complex<float> *>(pFFT->GetPtr()), pFFT->info.GetSize());

    auto &alg = GetAlg();
    ASSERT_TRUE(alg.Init(out, in, settings));
    ASSERT_TRUE(alg.Execute(out, in));
    // wait till the work is done
    alg.Synchronize();

    // Print(reinterpret_cast<std::complex<float> *>(pVolume->GetPtr()), pVolume->info.GetSize());

    TestResult(
      out, space, settings.GetType() == Settings::Type::kFast ? 0 : settings.GetBlobRadius());
  }


  std::unique_ptr<Payload<FourierDescriptor>> pFFT;
  std::unique_ptr<Payload<FourierDescriptor>> pVolume;
  std::unique_ptr<Payload<LogicalDescriptor>> pWeight;
  std::unique_ptr<Payload<LogicalDescriptor>> pTraverseSpace;
  std::unique_ptr<Payload<LogicalDescriptor>> pBlobTable;
};