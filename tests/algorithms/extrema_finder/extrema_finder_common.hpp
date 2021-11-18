#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <tests/algorithms/common.hpp>
#include <tests/utils.hpp>
#include <gmock/gmock.h>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;
using namespace umpalumpa::test;

class ExtremaFinder_Tests : public TestAlg<AExtremaFinder>
{
protected:
  auto CreatePayloadData(const Size &size)
  {
    auto ld = LogicalDescriptor(size);
    auto bytes = ld.Elems() * Sizeof(DataType::kFloat);
    auto pd = Create(bytes, DataType::kFloat);
    return Payload(ld, std::move(pd), "Data");
  }

  auto CreatePayloadValues(const Settings &settings, const Size &size)
  {
    auto ld = LogicalDescriptor(size);
    if (SearchResult::kValue != settings.GetResult()) {
      return Payload(ld, Create(0, DataType::kVoid), "Default (empty) Values");
    }
    auto bytes = ld.Elems() * Sizeof(DataType::kFloat);
    auto pd = Create(bytes, DataType::kFloat);
    return Payload(ld, std::move(pd), "Values");
  }

  auto CreatePayloadLocations(const Settings &settings, const Size &size)
  {
    auto ld = LogicalDescriptor(size);
    if (SearchResult::kLocation != settings.GetResult()) {
      return Payload(ld, Create(0, DataType::kVoid), "Default (empty) Locations");
    }
    auto bytes = ld.Elems() * Sizeof(DataType::kFloat);
    auto pd = Create(bytes, DataType::kFloat);
    return Payload(ld, std::move(pd), "Locations");
  }

  void SetUp(const Settings &settings, const Size &sizeData)
  {
    pData = std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadData(sizeData));
    Register(pData->dataInfo);

    auto sizeRes = Size(1, 1, 1, sizeData.n);
    pValues = std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadValues(settings, sizeRes));
    Register(pValues->dataInfo);

    pLocations =
      std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadLocations(settings, sizeRes));
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

  std::unique_ptr<Payload<LogicalDescriptor>> pValues;
  std::unique_ptr<Payload<LogicalDescriptor>> pLocations;
  std::unique_ptr<Payload<LogicalDescriptor>> pData;
};