#include <libumpalumpa/algorithms/fourier_transformation/fft_cpu.hpp>
#include <gtest/gtest.h>
#include <tests/algorithms/fourier_transformation/fft_tests.hpp>

using namespace umpalumpa::fourier_transformation;
using namespace umpalumpa::data;

class FFTCPUTest
  : public ::testing::Test
  , public FFT_Tests
{
public:
  void *Allocate(size_t bytes) override { return malloc(bytes); }

  void Free(void *ptr) override { free(ptr); }

  // CANNOT return "Free" method, because of the destruction order
  FreeFunction GetFree() override
  {
    return [](void *ptr) { free(ptr); };
  }

  FFTCPU &GetTransformer() override { return transformer; }

  ManagedBy GetManager() override { return ManagedBy::Manually; };

  int GetMemoryNode() override { return 0; }

protected:
  FFTCPU transformer = FFTCPU();
};
#define NAME FFTCPUTest
#include <tests/algorithms/fourier_transformation/afft_common.hpp>
