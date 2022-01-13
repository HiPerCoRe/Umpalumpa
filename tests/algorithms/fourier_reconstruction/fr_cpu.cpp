#include <tests/algorithms/fourier_reconstruction/afr_common.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cpu.hpp>

class FRCPUTest : public FR_Tests
{
public:
  FRCPU &GetAlg() override { return transformer; }

  using FR_Tests::SetUp;

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
  FRCPU transformer;
};

#define NAME FRCPUTest
#include <tests/algorithms/fourier_reconstruction/fr_tests.hpp>