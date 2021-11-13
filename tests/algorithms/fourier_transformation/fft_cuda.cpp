#include <tests/algorithms/fourier_transformation/afft_common.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>

class FFTCUDATest : public FFT_Tests
{
public:
  FFTCUDA &GetAlg() override { return transformer; }

  using FFT_Tests::SetUp;

  PhysicalDescriptor Create(size_t bytes, DataType type) override
  {
    void *ptr;
    CudaErrchk(cudaMallocManaged(&ptr, bytes));
    return PhysicalDescriptor(ptr, bytes, type, ManagedBy::CUDA, nullptr);
  }

  void Remove(const PhysicalDescriptor &pd) override { CudaErrchk(cudaFree(pd.GetPtr())); }

  void Register(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Unregister(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Acquire(const PhysicalDescriptor &pd) override
  {
    CudaErrchk(cudaMemPrefetchAsync(pd.GetPtr(), pd.GetBytes(), worker));
  }

  void Release(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

private:
  const int worker = 0;
  FFTCUDA transformer = FFTCUDA(worker);
};

#define NAME FFTCUDATest
#include <tests/algorithms/fourier_transformation/fft_tests.hpp>