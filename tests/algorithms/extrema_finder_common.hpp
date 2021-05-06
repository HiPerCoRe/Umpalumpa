#ifndef TESTS_ALGORITHMS_EXTREMA_FINDER_COMMON
#define TESTS_ALGORITHMS_EXTREMA_FINDER_COMMON
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <random>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;


template<typename T> void GenerateData(T *data, size_t elems)
{
  auto mt = std::mt19937(42);
  auto dist = std::normal_distribution<float>((float)0, (float)1);
  for (size_t i = 0; i < elems; ++i) { data[i] = dist(mt); }
}

template<typename T> void PrintData(std::unique_ptr<T> &data, const Size size)
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
  const auto ldIn = LogicalDescriptor(sizeIn, sizeIn, "IOTA");
  const auto dt = DataType::kFloat;
  const auto pdIn = PhysicalDescriptor(sizeIn.total * Sizeof(dt) + 13, dt);// add some extra bytes
  const auto in = SearchData(data.get(), ldIn, pdIn, "Random data");

  for (size_t i = 0; i < sizeIn.total; ++i) { ASSERT_FLOAT_EQ(data[i], i) << " for i=" << i; }

  const size_t batch = 3;
  const size_t full_batches = sizeIn.n / batch;
  for (size_t i = 0; i < full_batches; ++i) {
    const size_t startN = i * batch;
    const auto s = in.Subset(startN, batch);
    // check size
    ASSERT_EQ(s.info.size, Size(3, 5, 7, batch)) << " for i=" << i;
    const size_t offset = startN * sizeIn.single;
    // check that pointer is correct
    ASSERT_EQ(s.data, data.get() + offset) << " for i=" << i;
    // check that bytes are correct
    ASSERT_EQ(s.dataInfo.bytes, sizeIn.single * sizeof(float) * batch) << " for i=" << i;
  }
  // check last iteration
  const auto s = in.Subset(9, batch);
  // check size
  ASSERT_EQ(s.info.size, Size(3, 5, 7, 2));
  const size_t offset = 9 * sizeIn.single;
  // check that pointer is correct
  ASSERT_EQ(s.data, data.get() + offset);
  // check that bytes are correct
  ASSERT_EQ(s.dataInfo.bytes, sizeIn.single * sizeof(float) * 2);
}

TEST_F(NAME, 1D_batch_noPadd_max_valOnly)
{
  auto sizeIn = Size(10, 1, 1, 3);
  auto settings = Settings(SearchType::kMax, SearchLocation::kEntire, SearchResult::kValue);
  auto data = reinterpret_cast<float *>(allocate(sizeIn.total * sizeof(float)));
  auto dataOrig = std::unique_ptr<float[]>(new float[sizeIn.total]);
  auto ldIn = LogicalDescriptor(sizeIn, sizeIn, "Random input data");
  auto pdIn = PhysicalDescriptor(sizeIn.total * sizeof(float), DataType::kFloat);
  auto in = SearchData(data, ldIn, pdIn, "Random data");

  GenerateData(data, sizeIn.total);
  memcpy(dataOrig.get(), data, sizeIn.total * sizeof(float));
  //   PrintData(data, sizeIn);// FIXME add utility method to payload?

  auto sizeValues = Size(1, 1, 1, sizeIn.n);
  auto values = reinterpret_cast<float *>(allocate(sizeValues.total * sizeof(float)));
  auto ldVal = LogicalDescriptor(sizeValues, sizeValues, "Values of the found extremas");
  auto pdVal = PhysicalDescriptor(sizeValues.total * sizeof(float), DataType::kFloat);
  auto valuesP = Payload(values, ldVal, pdVal, "Resulting maxima");
  auto out = ResultData(&valuesP, nullptr);

  auto searcher = getSearcher();
  searcher.Init(out, in, settings);
  // make sure the settings is fine
  settings.dryRun = true;

  ASSERT_TRUE(searcher.Execute(out, in, settings));

  // make sure the search finished
  settings.dryRun = false;
  ASSERT_TRUE(searcher.Execute(out, in, settings));
  // make sure that we didn't change data
  ASSERT_EQ(0, memcmp(data, dataOrig.get(), sizeIn.total * sizeof(float)));
  // test that we found good maximas
  for (int i = 0; i < sizeValues.n; ++i) {
    auto first = data + (sizeIn.single * i);
    auto last = first + sizeIn.single;
    float trueMax = *std::max_element(first, last);
    ASSERT_FLOAT_EQ(trueMax, values[i]) << " for n=" << i;
  }
  free(data);
  free(values);
}

TEST_F(NAME, 1D_manyBatches_noPadd_max_valOnly)
{
  auto sizeIn = Size(120, 32, 16, 103);
  auto settings = Settings(SearchType::kMax, SearchLocation::kEntire, SearchResult::kValue);
  auto data = reinterpret_cast<float *>(allocate(sizeIn.total * sizeof(float)));
  auto dataOrig = std::unique_ptr<float[]>(new float[sizeIn.total]);
  auto ldIn = LogicalDescriptor(sizeIn, sizeIn, "Basic data");
  auto pdIn = PhysicalDescriptor(sizeIn.total * sizeof(float), DataType::kFloat);
  auto in = SearchData(data, ldIn, pdIn, "Random data");

  GenerateData(data, sizeIn.total);
  memcpy(dataOrig.get(), data, sizeIn.total * sizeof(float));
  //   PrintData(data, sizeIn);// FIXME add utility method to payload?

  auto sizeValues = Size(1, 1, 1, sizeIn.n);
  auto values = reinterpret_cast<float *>(allocate(sizeValues.total * sizeof(float)));
  auto ldVal = LogicalDescriptor(sizeValues, sizeValues, "Values of the found extremas");
  auto pdVal = PhysicalDescriptor(sizeValues.total * sizeof(float), DataType::kFloat);
  auto valuesP = Payload(values, ldVal, pdVal, "Result maxima");

  auto searcher = getSearcher();

  const size_t batchSize = 7;
  bool isFirstIter = true;
  for (size_t offset = 0; offset < sizeIn.n; offset += batchSize) {
    auto i = in.Subset(offset, batchSize);
    auto o = valuesP.Subset(offset, batchSize);
    auto out = ResultData(&o, nullptr);
    if (isFirstIter) {
      isFirstIter = false;
      searcher.Init(out, i, settings);
    }
    // make sure the settings is fine
    settings.dryRun = true;
    ASSERT_TRUE(searcher.Execute(out, i, settings));

    // make sure the search finished
    settings.dryRun = false;

    ASSERT_TRUE(searcher.Execute(out, i, settings));
  }

  // make sure that we didn't change data
  ASSERT_EQ(0, memcmp(data, dataOrig.get(), sizeIn.total * sizeof(float)));
  // test that we found good maximas
  for (int i = 0; i < sizeValues.n; ++i) {
    auto first = data + (sizeIn.single * i);
    auto last = first + sizeIn.single;
    float trueMax = *std::max_element(first, last);
    ASSERT_FLOAT_EQ(trueMax, values[i]) << " for n=" << i;
  }

  free(values);
  free(data);
}


#endif /* TESTS_ALGORITHMS_EXTREMA_FINDER_COMMON */
