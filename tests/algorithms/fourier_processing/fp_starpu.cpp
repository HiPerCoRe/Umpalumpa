#include <libumpalumpa/algorithms/fourier_processing/fp_starpu.hpp>
#include <starpu.h>
#include <gtest/gtest.h>
#include <tests/algorithms/fourier_processing/fp_tests.hpp>

using namespace umpalumpa::fourier_processing;
using namespace umpalumpa::data;
class FPStarPUTest
  : public ::testing::Test
  , public FP_Tests
{
public:
  FPStarPU &GetFourierProcessor() override { return transformer; }
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

  ManagedBy GetManager() override { return ManagedBy::StarPU; };

  int GetMemoryNode() override { return STARPU_MAIN_RAM; }

private:
  FPStarPU transformer;
};
#define NAME FPStarPUTest
#include <tests/algorithms/fourier_processing/afp_common.hpp>
