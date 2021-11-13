#include <tests/algorithms/fourier_processing/afp_common.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp>

class FPCUDATest : public FP_Tests
{
public:
  FPCUDA &GetFourierProcessor() override { return transformer; }

  using FP_Tests::SetUp;

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

  void Release(const PhysicalDescriptor &pd) override
  { /* nothing to do */
  }

private:
  const int worker = 0;
  FPCUDA transformer = FPCUDA(worker);
};

#define NAME FPCUDATest
#include <tests/algorithms/fourier_processing/fp_tests.hpp>