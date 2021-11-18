#include <tests/algorithms/fourier_transformation/afft_common.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_starpu.hpp>
#include <libumpalumpa/utils/starpu.hpp>

using umpalumpa::utils::StarPUUtils;

class FFTStarPUTest : public FFT_Tests
{
public:
  FFTStarPU &GetAlg() override { return transformer; }

  static void SetUpTestSuite() { STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "StarPU init"); }

  using FFT_Tests::SetUp;

  static void TearDownTestSuite() { starpu_shutdown(); }

  PhysicalDescriptor Create(size_t bytes, DataType type) override
  {
    void *ptr = nullptr;
    starpu_malloc(&ptr, bytes);
    memset(ptr, 0, bytes);
    auto *handle = new starpu_data_handle_t();
    return PhysicalDescriptor(ptr, bytes, type, ManagedBy::StarPU, handle);
  }

  void Remove(const PhysicalDescriptor &pd) override
  {
    starpu_free(pd.GetPtr());
    delete StarPUUtils::GetHandle(pd);
  }

  void Register(const PhysicalDescriptor &pd) override { StarPUUtils::Register(pd); };

  void Unregister(const PhysicalDescriptor &pd) override { StarPUUtils::Unregister(pd); };

  void Acquire(const PhysicalDescriptor &pd) override
  {
    starpu_data_acquire(*StarPUUtils::GetHandle(pd), STARPU_RW);
  }

  void Release(const PhysicalDescriptor &pd) override
  {
    starpu_data_release(*StarPUUtils::GetHandle(pd));
  }

private:
  FFTStarPU transformer;
};

#define NAME FFTStarPUTest
#include <tests/algorithms/fourier_transformation/fft_tests.hpp>