#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_starpu.hpp>
#include <starpu.h>
#include <gtest/gtest.h>
using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;

class SingleExtremaFinderStarPUTest : public ::testing::Test
{
public:
  auto &GetSearcher() { return SingleExtremaFinderStarPU::Instance(); }
  void SetUp() override { STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "StarPU init"); }

  void TearDown() override { starpu_shutdown(); }

  void WaitTillDone()
  {
    STARPU_CHECK_RETURN_VALUE(starpu_task_wait_for_all(), "Waiting for all tasks");
  }

  auto Allocate(size_t bytes) { return malloc(bytes); }
  auto Free(void *ptr) { free(ptr); }
};
#define NAME SingleExtremaFinderStarPUTest
#include <tests/algorithms/extrema_finder_common.hpp>