#pragma once

#include <libumpalumpa/operations/extrema_finder/aextrema_finder.hpp>
#include <tests/operations/common.hpp>
#include <tests/utils.hpp>
#include <gmock/gmock.h>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;
using namespace umpalumpa::test;

class ExtremaFinder_Tests : public TestOp<AExtremaFinder>
{
protected:
  auto CreatePayloadData(const Size &size)
  {
    auto ld = LogicalDescriptor(size);
    auto type = DataType::Get<float>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Data");
  }

  auto CreatePayloadValues(const Settings &settings, const Size &size)
  {
    auto ld = LogicalDescriptor(size);
    if (Result::kValue != settings.GetResult()) {
      return Payload(ld, Create(0, DataType::Get<void>()), "Default (empty) Values");
    }
    auto type = DataType::Get<float>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Values");
  }

  auto CreatePayloadLocations(const Settings &settings, const Size &size)
  {
    auto ld = LogicalDescriptor(size);
    if (Result::kLocation != settings.GetResult()) {
      return Payload(ld, Create(0, DataType::Get<void>()), "Default (empty) Locations");
    }
    auto type = DataType::Get<float>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Locations");
  }

  void SetUp(const Settings &settings, const Size &sizeData)
  {
    pData = std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadData(sizeData));
    Register(pData->dataInfo);

    auto sizeVals = Size(1, 1, 1, sizeData.n);
    pValues = std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadValues(settings, sizeVals));
    Register(pValues->dataInfo);

    auto sizeLocs = Size(sizeData.GetDimAsNumber(), 1, 1, sizeData.n);
    pLocations =
      std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadLocations(settings, sizeLocs));
    Register(pLocations->dataInfo);
  }

  /**
   * Called at the end of each test fixture
   **/
  void TearDown() override
  {
    Unregister(pValues->dataInfo);
    Remove(pValues->dataInfo);

    Unregister(pLocations->dataInfo);
    Remove(pLocations->dataInfo);

    Unregister(pData->dataInfo);
    Remove(pData->dataInfo);
  }

  void CheckValues()
  {
    Acquire(pData->dataInfo);
    Acquire(pValues->dataInfo);
    auto &size = pData->info.GetSize();
    for (int n = 0; n < size.n; ++n) {
      auto first = reinterpret_cast<float *>(pData->GetPtr()) + (size.single * n);
      auto last = first + size.single;
      const float trueMax = *std::max_element(first, last);
      const float foundMax = reinterpret_cast<float *>(pValues->GetPtr())[n];
      ASSERT_THAT(foundMax, ::testing::NanSensitiveFloatEq(trueMax)) << " for n=" << n;
    }
    Release(pValues->dataInfo);
    Release(pData->dataInfo);
  }

  void CheckLocationsSinglePrecision()
  {
    Acquire(pData->dataInfo);
    Acquire(pLocations->dataInfo);
    const auto &sizeData = pData->info.GetSize();
    // test that we found good maximas
    for (size_t n = 0; n < sizeData.n; ++n) {
      auto *start = reinterpret_cast<float *>(pData->GetPtr()) + n * sizeData.single;
      auto max = start[0];
      size_t expectedX = 0;
      size_t expectedY = 0;
      size_t expectedZ = 0;
      for (size_t z = 0; z < sizeData.z; ++z) {
        for (size_t y = 0; y < sizeData.y; ++y) {
          for (size_t x = 0; x < sizeData.x; ++x) {
            auto v = start[z * sizeData.y * sizeData.x + y * sizeData.x + x];
            if (v > max) {
              max = v;
              expectedX = x;
              expectedY = y;
              expectedZ = z;
            }
          }
        }
      }
      auto *actualLoc =
        reinterpret_cast<float *>(pLocations->GetPtr()) + n * pLocations->info.GetSize().single;
      ASSERT_EQ(expectedX, actualLoc[0]) << " for n=" << n;
      if (sizeData.GetDimAsNumber() > 1) {
        ASSERT_EQ(expectedY, actualLoc[1]) << " for n=" << n;
        if (sizeData.GetDimAsNumber() > 2) { ASSERT_EQ(expectedZ, actualLoc[2]) << " for n=" << n; }
      }
    }
    Release(pLocations->dataInfo);
    Release(pData->dataInfo);
  }

  void TestLocsMaxEntireSingle(const Size &size)
  {
    auto settings =
      Settings(ExtremaType::kMax, Location::kEntire, Result::kLocation, Precision::kSingle);

    SetUp(settings, size);

    Acquire(pData->dataInfo);
    FillNormalDist(reinterpret_cast<float *>(pData->GetPtr()), pData->info.GetSize().total);
    Release(pData->dataInfo);

    auto out = AExtremaFinder::OutputData(*pValues, *pLocations);
    auto in = AExtremaFinder::InputData(*pData);

    auto &op = GetOp();
    ASSERT_TRUE(op.Init(out, in, settings));
    ASSERT_TRUE(op.Execute(out, in));
    // wait till the work is done
    op.Synchronize();

    // check results
    CheckLocationsSinglePrecision();
  }

  std::unique_ptr<Payload<LogicalDescriptor>> pValues;
  std::unique_ptr<Payload<LogicalDescriptor>> pLocations;
  std::unique_ptr<Payload<LogicalDescriptor>> pData;
};
