#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <gtest/gtest.h>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;

class SingleExtremaFinderCPUTest : public ::testing::Test
{
public:
  auto getSearcher() { return SingleExtremaFinderCPU(); }
  auto allocate(size_t bytes)
  {
    return malloc(bytes);
  }
};
#define NAME SingleExtremaFinderCPUTest
#include <tests/algorithms/extrema_finder_common.hpp>