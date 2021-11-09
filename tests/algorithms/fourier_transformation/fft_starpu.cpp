#include <libumpalumpa/algorithms/fourier_transformation/fft_starpu.hpp>
#include <starpu.h>
#include <gtest/gtest.h>
#include <tests/algorithms/fourier_transformation/fft_tests.hpp>

using namespace umpalumpa::fourier_transformation;
using namespace umpalumpa::data;

class FFTStarPUTest
  : public ::testing::Test
  , public FFT_Tests
{
public:
  FFTStarPU &GetTransformer() override { return transformer; }
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
  FFTStarPU transformer;
};
#define NAME FFTStarPUTest
#include <tests/algorithms/fourier_transformation/afft_common.hpp>
