#include <tests/algorithms/correlation/acorrelation_common.hpp>
#include <libumpalumpa/algorithms/correlation/correlation_cpu.hpp>
#include <algorithm>

class CorrelationCPUTest : public Correlation_Tests
{
public:
  Correlation_CPU &GetAlg() override { return transformer; }

  using Correlation_Tests::SetUp;

  PhysicalDescriptor Create(size_t bytes, DataType type) override
  {
    auto *ptr = malloc(bytes);
    memory.emplace_back(ptr);
    return PhysicalDescriptor(malloc(bytes), bytes, type, ManagedBy::Manually, nullptr);
  }

  PhysicalDescriptor Copy(const PhysicalDescriptor &pd) override
  {
    return pd.CopyWithPtr(pd.GetPtr());
  }

  void Remove(const PhysicalDescriptor &pd) override
  {
    if (auto it = std::find(memory.begin(), memory.end(), pd.GetPtr()); memory.end() != it) {
      free(pd.GetPtr());
      memory.erase(it);
    }
  }

  void Register(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Unregister(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Acquire(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Release(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

private:
  Correlation_CPU transformer;
  std::vector<void *> memory;
};

#define NAME CorrelationCPUTest
#include <tests/algorithms/correlation/correlation_tests.hpp>