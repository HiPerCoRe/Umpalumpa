#include <tests/operations/fourier_reconstruction/afr_common.hpp>
#include <libumpalumpa/operations/fourier_reconstruction/fr_cuda.hpp>

class FRCUDATest : public FR_Tests
{
public:
  FRCUDA &GetOp() override { return transformer; }

  using FR_Tests::SetUp;

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
  FRCUDA transformer = FRCUDA(worker);
};

#define NAME FRCUDATest
#include <tests/operations/fourier_reconstruction/fr_tests.hpp>