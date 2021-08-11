#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <gtest/gtest.h>
#include <libumpalumpa/utils/cuda.hpp>
#include <tests/algorithms/fourier_transformation/fft_tests.hpp>

#include <cuda_runtime.h>

using namespace umpalumpa::fourier_transformation;
using namespace umpalumpa::data;

class FFTCUDATest : public ::testing::Test, public FFT_Tests
{
public:
  auto Allocate(size_t bytes)
  {
    void *ptr;
    CudaErrchk(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  void Free(void *ptr) { cudaFree(ptr); }

  FFTCUDA &GetTransformer() override { return transformer; }

private:
  FFTCUDA transformer = FFTCUDA(0);
};
#define NAME FFTCUDATest
#include <tests/algorithms/fourier_transformation/afft_common.hpp>

