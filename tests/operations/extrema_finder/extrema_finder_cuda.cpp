#include <tests/operations/extrema_finder/extrema_finder_common.hpp>
#include <libumpalumpa/operations/extrema_finder/single_extrema_finder_cuda.hpp>

class SingleExtremaFinderCUDATest : public ExtremaFinder_Tests
{
public:
  SingleExtremaFinderCUDA &GetOp() override { return transformer; }

  using ExtremaFinder_Tests::SetUp;

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
  SingleExtremaFinderCUDA transformer = SingleExtremaFinderCUDA(worker);
};

#define NAME SingleExtremaFinderCUDATest
#include <tests/operations/extrema_finder/extrema_finder_tests.hpp>