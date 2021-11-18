#include <tests/algorithms/correlation/acorrelation_common.hpp>
#include <libumpalumpa/algorithms/correlation/correlation_cuda.hpp>
#include <algorithm>

class CorrelationCUDATest : public Correlation_Tests
{
public:
  Correlation_CUDA &GetAlg() override { return transformer; }

  using Correlation_Tests::SetUp;

  PhysicalDescriptor Create(size_t bytes, DataType type) override
  {
    void *ptr;
    CudaErrchk(cudaMallocManaged(&ptr, bytes));
    memset(ptr, 0, bytes);
    memory.emplace_back(ptr);
    return PhysicalDescriptor(ptr, bytes, type, ManagedBy::CUDA, nullptr);
  }

  PhysicalDescriptor Copy(const PhysicalDescriptor &pd) override
  {
    return pd.CopyWithPtr(pd.GetPtr());
  }

  void Remove(const PhysicalDescriptor &pd) override
  {
    if (auto it = std::find(memory.begin(), memory.end(), pd.GetPtr()); memory.end() != it) {
      CudaErrchk(cudaFree(pd.GetPtr()));
      memory.erase(it);
    }
  }

  void Register(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Unregister(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Acquire(const PhysicalDescriptor &pd) override
  {
    CudaErrchk(cudaMemPrefetchAsync(pd.GetPtr(), pd.GetBytes(), worker));
  }

  void Release(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

private:
  const int worker = 0;
  Correlation_CUDA transformer = Correlation_CUDA(worker);
  std::vector<void *> memory;
};

#define NAME CorrelationCUDATest
#include <tests/algorithms/correlation/correlation_tests.hpp>