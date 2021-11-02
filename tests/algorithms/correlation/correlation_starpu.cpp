#include <libumpalumpa/algorithms/correlation/correlation_starpu.hpp>
#include <starpu.h>
#include <gtest/gtest.h>
#include <tests/algorithms/correlation/correlation_tests.hpp>

using namespace umpalumpa::correlation;
using namespace umpalumpa::data;
class CorrelationStarPUTest
  : public ::testing::Test
  , public Correlation_Tests
{
public:
  CorrelationStarPU &GetTransformer() override { return transformer; }
  void SetUp() override { STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "StarPU init"); }

  void TearDown() override { starpu_shutdown(); }

  void WaitTillDone()
  {
    STARPU_CHECK_RETURN_VALUE(starpu_task_wait_for_all(), "Waiting for all tasks");
  }

  void *Allocate(size_t bytes)
  {
    void *ptr = nullptr;
    starpu_malloc(&ptr, bytes);
    return ptr;
  }
  void Free(void *ptr) { starpu_free(ptr); }

  FreeFunction GetFree() override
  {
    return [](void *ptr) { starpu_free(ptr); };
  }

private:
  CorrelationStarPU transformer;
};
#define NAME CorrelationStarPUTest
#include <tests/algorithms/correlation/acorrelation_common.hpp>
