#include <tests/operations/initialization/common.hpp>
#include <libumpalumpa/operations/initialization/cpu.hpp>

class InitializationCPUTest : public Initialization_Tests
{
public:
  CPU &GetOp() override { return op; }

  using Initialization_Tests::SetUp;

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

#define NAME InitializationCPUTest
#include <tests/operations/initialization/tests.hpp>