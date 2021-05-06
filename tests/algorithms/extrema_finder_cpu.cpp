#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <gtest/gtest.h>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;

class SingleExtremaFinderCPUTest : public ::testing::Test
{
public:
  auto &GetSearcher() { return searcher; }
  auto Allocate(size_t bytes) { return malloc(bytes); }
  auto Free(void *ptr) { free(ptr); }
  void WaitTillDone(){};

private:
  SingleExtremaFinderCPU searcher;
};
#define NAME SingleExtremaFinderCPUTest
#include <tests/algorithms/extrema_finder_common.hpp>