#include <libumpalumpa/algorithms/correlation/correlation_cuda.hpp>
#include <gtest/gtest.h>
#include <libumpalumpa/utils/cuda.hpp>
#include <tests/algorithms/correlation/correlation_tests.hpp>

#include <cuda_runtime.h>

using namespace umpalumpa::correlation;
using namespace umpalumpa::data;

class CorrelationCUDATest : public ::testing::Test, public Correlation_Tests 
{
public:
  void *Allocate(size_t bytes) override {
    void *ptr;
    CudaErrchk(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  void Free(void *ptr) override { cudaFree(ptr); }

  // CANNOT return "Free" method, because of the destruction order
  FreeFunction GetFree() override { return [](void *ptr){ CudaErrchk(cudaFree(ptr));}; }

  Correlation_CUDA &GetTransformer() override { return transformer; }

protected:
  Correlation_CUDA transformer = Correlation_CUDA(0);
};
#define NAME CorrelationCUDATest
#include <tests/algorithms/correlation/acorrelation_common.hpp>

