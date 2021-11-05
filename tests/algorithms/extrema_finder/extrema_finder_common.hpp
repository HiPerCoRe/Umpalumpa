#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>
#include <memory>
#include <random>
#include <fcntl.h>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;


template<typename T> void GenerateData(T *data, size_t elems)
{
  auto mt = std::mt19937(42);
  auto dist = std::normal_distribution<float>((float)0, (float)1);
  for (size_t i = 0; i < elems; ++i) { data[i] = dist(mt); }
}

template<typename T> void FillRandomBytes(T *dst, size_t bytes)
{
  int fd = open("/dev/urandom", O_RDONLY);
  read(fd, dst, bytes);
}

template<typename T> void PrintData(T *data, const Size size)
{
  for (size_t n = 0; n < size.n; ++n) {
    size_t offset = n * size.single;
    for (size_t i = 0; i < size.single; ++i) { printf("%+.3f\t", data[offset + i]); }
    std::cout << "\n";
  }
}

TEST_F(NAME, SearchData_Subset)
{
  const auto sizeIn = Size(3, 5, 7, 11);
  const auto data = std::unique_ptr<float[]>(new float[sizeIn.total]);
  std::iota(data.get(), data.get() + sizeIn.total, 0);
  const auto ldIn = LogicalDescriptor(sizeIn);
  const auto dt = DataType::kFloat;
  const auto pdIn =
    PhysicalDescriptor(data.get(), sizeIn.total * Sizeof(dt) + 13, dt);// add some extra bytes
  const auto in = AExtremaFinder::InputData(Payload(ldIn, pdIn, "Random data"));

  for (size_t i = 0; i < sizeIn.total; ++i) { ASSERT_FLOAT_EQ(data[i], i) << " for i=" << i; }

  const size_t batch = 3;
  const size_t full_batches = sizeIn.n / batch;
  for (size_t i = 0; i < full_batches; ++i) {
    const size_t startN = i * batch;
    const auto s = in.GetData().Subset(startN, batch);
    // check size
    ASSERT_EQ(s.info.GetSize(), Size(3, 5, 7, batch)) << " for i=" << i;
    const size_t offset = startN * sizeIn.single;
    // check that pointer is correct
    ASSERT_EQ(s.GetPtr(), data.get() + offset) << " for i=" << i;
    // check that bytes are correct
    ASSERT_EQ(s.dataInfo.GetBytes(), sizeIn.single * sizeof(float) * batch) << " for i=" << i;
  }
  // check last iteration
  const auto s = in.GetData().Subset(9, batch);
  // check size
  ASSERT_EQ(s.info.GetSize(), Size(3, 5, 7, 2));
  const size_t offset = 9 * sizeIn.single;
  // check that pointer is correct
  ASSERT_EQ(s.GetPtr(), data.get() + offset);
  // check that bytes are correct
  ASSERT_EQ(s.dataInfo.GetBytes(), sizeIn.single * sizeof(float) * 2);
}

TEST_F(NAME, 1D_batch_noPadd_max_valOnly)
{
  auto sizeIn = Size(10, 1, 1, 3);
  auto settings = Settings(SearchType::kMax, SearchLocation::kEntire, SearchResult::kValue);
  auto data = reinterpret_cast<float *>(Allocate(sizeIn.total * sizeof(float)));
  auto dataOrig = std::unique_ptr<float[]>(new float[sizeIn.total]);
  auto ldIn = LogicalDescriptor(sizeIn);
  auto pdIn = PhysicalDescriptor(data, sizeIn.total * sizeof(float), DataType::kFloat);
  auto in = AExtremaFinder::InputData(Payload(ldIn, pdIn, "Random data"));

  GenerateData(data, sizeIn.total);
  memcpy(dataOrig.get(), data, sizeIn.total * sizeof(float));
  // in.GetData().PrintData(std::cout, sizeIn);

  auto sizeValues = Size(1, 1, 1, sizeIn.n);
  auto values = reinterpret_cast<float *>(Allocate(sizeValues.total * sizeof(float)));
  auto ldVal = LogicalDescriptor(sizeValues);
  auto pdVal = PhysicalDescriptor(values, sizeValues.total * sizeof(float), DataType::kFloat);
  auto valuesP = Payload(ldVal, pdVal, "Resulting maxima");
  auto out = AExtremaFinder::OutputData::ValuesOnly(valuesP);

  AExtremaFinder &searcher = GetSearcher();
  ASSERT_TRUE(searcher.Init(out, in, settings));// including data, on purpose

  // make sure the search finished
  ASSERT_TRUE(searcher.Execute(out, in));

  WaitTillDone();

  // make sure that we didn't change data
  ASSERT_EQ(0, memcmp(data, dataOrig.get(), sizeIn.total * sizeof(float)));
  // test that we found good maximas
  // FIXME this comparison is very costly, run it in parallel
  for (int n = 0; n < sizeValues.n; ++n) {
    auto first = data + (sizeIn.single * n);
    auto last = first + sizeIn.single;
    const float trueMax = *std::max_element(first, last);
    ASSERT_THAT(values[n], ::testing::NanSensitiveFloatEq(trueMax)) << " for n=" << n;
  }
  Free(data);
  Free(values);
}

TEST_F(NAME, 3D_manyBatches_noPadd_max_valOnly)
{
  auto sizeIn = Size(120, 173, 150, 103);
  std::cout << "This test will need at least " << sizeIn.total * sizeof(float) / 1048576 << " MB"
            << std::endl;
  auto settings = Settings(SearchType::kMax, SearchLocation::kEntire, SearchResult::kValue);
  auto data = reinterpret_cast<float *>(Allocate(sizeIn.total * sizeof(float)));
  auto ldIn = LogicalDescriptor(sizeIn);
  auto pdIn = PhysicalDescriptor(data, sizeIn.total * sizeof(float), DataType::kFloat);
  auto inP = Payload(ldIn, pdIn, "Random data");

  FillRandomBytes(data, sizeIn.total * sizeof(float));

  auto sizeValues = Size(1, 1, 1, sizeIn.n);
  auto values = reinterpret_cast<float *>(Allocate(sizeValues.total * sizeof(float)));
  auto ldVal = LogicalDescriptor(sizeValues);
  auto pdVal = PhysicalDescriptor(values, sizeValues.total * sizeof(float), DataType::kFloat);
  auto valuesP = Payload(ldVal, pdVal, "Result maxima");

  AExtremaFinder &searcher = GetSearcher();

  const size_t batchSize = 134;
  bool isFirstIter = true;
  for (size_t offset = 0; offset < sizeIn.n; offset += batchSize) {
    auto i = inP.Subset(offset, batchSize);
    auto in = AExtremaFinder::InputData(std::move(i));
    auto o = valuesP.Subset(offset, batchSize);
    auto out = AExtremaFinder::OutputData::ValuesOnly(std::move(o));
    if (isFirstIter) {
      isFirstIter = false;
      ASSERT_TRUE(searcher.Init(out, in, settings));
    }
    ASSERT_TRUE(searcher.Execute(out, in));
  }

  WaitTillDone();

  // test that we found good maximas
  for (int n = 0; n < sizeValues.n; ++n) {
    auto first = data + (sizeIn.single * n);
    auto last = first + sizeIn.single;
    float trueMax = *std::max_element(first, last);
    ASSERT_THAT(values[n], ::testing::NanSensitiveFloatEq(trueMax)) << " for n=" << n;
  }

  Free(values);
  Free(data);
}

TEST_F(NAME, 2D_batch_noPadd_max_rectCenter_posOnly)
{
  auto sizeIn = Size(120, 100, 1, 3);

  auto settings = Settings(SearchType::kMax, SearchLocation::kRectCenter, SearchResult::kLocation);
  auto data = reinterpret_cast<float *>(Allocate(sizeIn.total * sizeof(float)));
  auto ldIn = LogicalDescriptor(sizeIn);
  auto pdIn = PhysicalDescriptor(data, sizeIn.total * sizeof(float), DataType::kFloat);
  auto inP = Payload(ldIn, pdIn, "Random data");

  for (size_t i = 0; i < sizeIn.total; i++) { data[i] = i; }

  auto sizeValues = Size(1, 1, 1, sizeIn.n);
  auto locations = reinterpret_cast<float *>(Allocate(sizeValues.total * sizeof(float)));
  auto ldVal = LogicalDescriptor(sizeValues);
  auto pdVal = PhysicalDescriptor(locations, sizeValues.total * sizeof(float), DataType::kFloat);
  auto locationsP = Payload(ldVal, pdVal, "Result maxima");

  AExtremaFinder &searcher = GetSearcher();

  auto in = AExtremaFinder::InputData(std::move(inP));
  auto out = AExtremaFinder::OutputData::LocationsOnly(std::move(locationsP));

  ASSERT_TRUE(searcher.Init(out, in, settings));
  ASSERT_TRUE(searcher.Execute(out, in));

  WaitTillDone();

  // FIXME these values should be read from settings
  Size searchRect(28, 17, 1, 1);
  size_t searchRectOffsetX = (in.GetData().info.GetPaddedSize().x - searchRect.x) / 2;
  size_t searchRectOffsetY = (in.GetData().info.GetPaddedSize().y - searchRect.y) / 2;
  // test that we found good maximas
  for (int n = 0; n < sizeValues.n; ++n) {
    auto expectedResult =
      searchRectOffsetY * sizeIn.x + searchRectOffsetX;// beginning of searchRect
    expectedResult +=
      sizeIn.x * (searchRect.y - 1) + (searchRect.x - 1);// last element of searchRect
    ASSERT_FLOAT_EQ(expectedResult, locations[n]) << " for n=" << n;
  }

  Free(locations);
  Free(data);
}
