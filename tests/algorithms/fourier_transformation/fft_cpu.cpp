#include <tests/algorithms/fourier_transformation/afft_common.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_cpu.hpp>

class FFTCPUTest : public FFT_Tests
{
public:
  FFTCPU &GetAlg() override { return transformer; }

  using FFT_Tests::SetUp;

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
  FFTCPU transformer;
};

#define NAME FFTCPUTest
#include <tests/algorithms/fourier_transformation/fft_tests.hpp>