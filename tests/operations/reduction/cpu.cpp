#include <tests/operations/reduction/common.hpp>
#include <libumpalumpa/operations/reduction/cpu.hpp>

class ReductionCPUTest : public Reduction_Tests
{
public:
  CPU &GetOp() override { return op; }

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
  CPU op;
};

#define NAME ReductionCPUTest
#include <tests/operations/reduction/tests.hpp>