#include <tests/algorithms/extrema_finder/extrema_finder_common.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>

class SingleExtremaFinderCPUTest : public ExtremaFinder_Tests
{
public:
  SingleExtremaFinderCPU &GetAlg() override { return transformer; }

  using ExtremaFinder_Tests::SetUp;

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
  SingleExtremaFinderCPU transformer;
};

#define NAME SingleExtremaFinderCPUTest
#include <tests/algorithms/extrema_finder/extrema_finder_tests.hpp>