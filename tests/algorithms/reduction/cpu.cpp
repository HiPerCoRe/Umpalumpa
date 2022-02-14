#include <tests/algorithms/reduction/common.hpp>
#include <libumpalumpa/algorithms/reduction/cpu.hpp>

class ReductionCPUTest : public Reduction_Tests
{
public:
  CPU &GetAlg() override { return alg; }

  using Reduction_Tests::SetUp;

  PhysicalDescriptor Create(size_t bytes, DataType type) override
  {
    return PhysicalDescriptor(
      (0 == bytes) ? nullptr : malloc(bytes), bytes, type, ManagedBy::Manually, nullptr);
  }

  void Remove(const PhysicalDescriptor &pd) override { free(pd.GetPtr()); }

  void Register(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Unregister(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Acquire(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Release(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

private:
  CPU alg;
};

#define NAME ReductionCPUTest
#include <tests/algorithms/reduction/tests.hpp>