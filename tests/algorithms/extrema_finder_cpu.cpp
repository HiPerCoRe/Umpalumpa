#include <gtest/gtest.h>

#include <iostream>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <memory>
#include <random>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;

template<typename T> void GenerateData(std::unique_ptr<T> &data, size_t elems)
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

TEST(ExtermaFinderCPU, 1D_batch_noPadd_max_valOnly)
{
  auto sizeIn = Size(10, 1, 1, 3);
  auto settings = Settings(SearchType::kMax, SearchLocation::kEntire, SearchResult::kValue);
  auto data = std::unique_ptr<float[]>(new float[sizeIn.total]);
  auto dataOrig = std::unique_ptr<float[]>(new float[sizeIn.total]);
  auto in = SearchData(data.get(), { sizeIn, sizeIn }, { sizeIn.total * sizeof(float), DataType::kFloat });

  GenerateData(data, sizeIn.total);
  memcpy(dataOrig.get(), data.get(), sizeIn.total * sizeof(float));
  //   PrintData(data, sizeIn);// FIXME add utility method to payload?

  auto sizeValues = Size(1, 1, 1, sizeIn.n);
  auto values = std::unique_ptr<float[]>(new float[sizeValues.total]);
  auto valuesP =
    Payload(values.get(), { sizeValues, sizeValues }, { sizeValues.total * sizeof(float), DataType::kFloat });

  auto searcher = SingleExtremaFinderCPU();
  // make sure the settings is fine
  settings.dryRun = true;
  ASSERT_TRUE(searcher.Execute({ &valuesP, nullptr }, in, settings));

  // make sure the search finished
  settings.dryRun = false;
  ASSERT_TRUE(searcher.Execute({ &valuesP, nullptr }, in, settings));
  // make sure that we didn't change data
  ASSERT_EQ(0, memcmp(data.get(), dataOrig.get(), sizeIn.total * sizeof(float)));
  // test that we found good maximas
  for (int i = 0; i < sizeValues.n; ++i) {
    auto first = data.get() + (sizeIn.single * i);
    auto last = first + sizeIn.single;
    float trueMax = *std::max_element(first, last);
    ASSERT_FLOAT_EQ(trueMax, values[i]) << " for n=" << i;
  }
}
