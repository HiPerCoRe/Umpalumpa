#include <tests/algorithms/fourier_processing/afp_common.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu.hpp>

class FPCPUTest : public FP_Tests
{
public:
  FPCPU &GetAlg() override { return transformer; }

  using FP_Tests::SetUp;

  PhysicalDescriptor Create(size_t bytes, DataType type) override
  {
    return PhysicalDescriptor(malloc(bytes), bytes, type, ManagedBy::Manually, nullptr);
  }

  void Remove(const PhysicalDescriptor &pd) override { free(pd.GetPtr()); }

  void Register(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Unregister(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Acquire(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Release(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

private:
  FPCPU transformer;
};

#define NAME FPCPUTest
#include <tests/algorithms/fourier_processing/fp_tests.hpp>