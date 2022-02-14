#include <tests/algorithms/reduction/common.hpp>
#include <libumpalumpa/algorithms/reduction/cuda.hpp>

class ReductionCUDATest : public Reduction_Tests
{
public:
  CUDA &GetAlg() override { return alg; }

  using Reduction_Tests::SetUp;

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
  CUDA alg = CUDA(worker);
};

#define NAME ReductionCUDATest
#include <tests/algorithms/reduction/tests.hpp>