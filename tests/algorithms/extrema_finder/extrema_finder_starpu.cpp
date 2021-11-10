#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_starpu.hpp>
#include <starpu.h>
#include <gtest/gtest.h>
using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;

class SingleExtremaFinderStarPUTest : public ::testing::Test
{
public:
  auto &GetSearcher() { return searcher; }
  void SetUp() override { STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "StarPU init"); }

  void TearDown() override { starpu_shutdown(); }

  void WaitTillDone()
  {
    STARPU_CHECK_RETURN_VALUE(starpu_task_wait_for_all(), "Waiting for all tasks");
  }

  auto *Allocate(size_t bytes)
  {
    void *ptr = nullptr;
    starpu_malloc(&ptr, bytes);
    return ptr;
  }
  auto Free(void *ptr) { starpu_free(ptr); }

  ManagedBy GetManager() { return ManagedBy::StarPU; };

  int GetMemoryNode() { return STARPU_MAIN_RAM; }

private:
  SingleExtremaFinderStarPU searcher;
};
#define NAME SingleExtremaFinderStarPUTest
#include <tests/algorithms/extrema_finder/extrema_finder_common.hpp>
