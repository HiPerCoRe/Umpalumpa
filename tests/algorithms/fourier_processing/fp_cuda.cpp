#include <libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp>
#include <gtest/gtest.h>
#include <libumpalumpa/utils/cuda.hpp>
#include <tests/algorithms/fourier_processing/fp_tests.hpp>

#include <cuda_runtime.h>

using namespace umpalumpa::fourier_processing;
using namespace umpalumpa::data;

class FPCUDATest
  : public ::testing::Test
  , public FP_Tests
{
public:
  void *Allocate(size_t bytes) override
  {
    void *ptr;
    CudaErrchk(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  void Free(void *ptr) override { cudaFree(ptr); }

  // CANNOT return "Free" method, because of the destruction order
  FreeFunction GetFree() override
  {
    return [](void *ptr) { CudaErrchk(cudaFree(ptr)); };
  }

  FPCUDA &GetFourierProcessor() override { return transformer; }

  ManagedBy GetManager() override { return ManagedBy::CUDA; };

  int GetMemoryNode() override { return 0; }

protected:
  FPCUDA transformer = FPCUDA(0);
};
#define NAME FPCUDATest
#include <tests/algorithms/fourier_processing/afp_common.hpp>
