#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_starpu.hpp>
#include <starpu.h>
#include <iostream>
#include <gtest/gtest.h>
using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;


auto getSearcher() { return SingleExtremaFinderStarPU(); }

class SingleExtremaFinderStarPUTest : public ::testing::Test
{
  void SetUp() override
  {
    std::cout << "SetUp()\n";
    starpu_init(NULL);
  }

  void TearDown() override
  {
    std::cout << "TearDown()\n";
    starpu_shutdown();
  }
};
#define NAME SingleExtremaFinderStarPUTest
#include <tests/algorithms/extrema_finder_common.hpp>