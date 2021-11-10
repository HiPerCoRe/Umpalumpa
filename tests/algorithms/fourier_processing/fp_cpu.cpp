#include <libumpalumpa/algorithms/fourier_processing/fp_cpu.hpp>
#include <gtest/gtest.h>
#include <libumpalumpa/utils/cuda.hpp>
#include <tests/algorithms/fourier_processing/fp_tests.hpp>

#include <cuda_runtime.h>

using namespace umpalumpa::fourier_processing;
using namespace umpalumpa::data;

class FPCPUTest
  : public ::testing::Test
  , public FP_Tests
{
public:
  void *Allocate(size_t bytes) override { return malloc(bytes); }

  void Free(void *ptr) override { free(ptr); }

  // CANNOT return "Free" method, because of the destruction order
  FreeFunction GetFree() override
  {
    return [](void *ptr) { free(ptr); };
  }

  FPCPU &GetFourierProcessor() override { return fourierProcessor; }

  ManagedBy GetManager() override { return ManagedBy::Manually; };

  int GetMemoryNode() override { return 0; }

protected:
  FPCPU fourierProcessor;
};
#define NAME FPCPUTest
#include <tests/algorithms/fourier_processing/afp_common.hpp>
