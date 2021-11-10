#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <gtest/gtest.h>
#include <tests/algorithms/fourier_transformation/fft_tests.hpp>

#include <cuda_runtime.h>

using namespace umpalumpa::fourier_transformation;
using namespace umpalumpa::data;

class FFTCUDATest
  : public ::testing::Test
  , public FFT_Tests
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

  FFTCUDA &GetTransformer() override { return transformer; }

  ManagedBy GetManager() override { return ManagedBy::CUDA; };

  int GetMemoryNode() override { return 0; }

protected:
  FFTCUDA transformer = FFTCUDA(0);
};
#define NAME FFTCUDATest
#include <tests/algorithms/fourier_transformation/afft_common.hpp>
