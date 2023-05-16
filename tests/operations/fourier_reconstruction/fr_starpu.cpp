#include <tests/operations/fourier_reconstruction/afr_common.hpp>
#include <libumpalumpa/operations/fourier_reconstruction/fr_starpu.hpp>
#include <libumpalumpa/utils/starpu.hpp>

using umpalumpa::utils::StarPUUtils;

class FRStarPUTest : public FR_Tests
{
public:
  FRStarPU &GetOp() override { return transformer; }

  static void SetUpTestSuite() { STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "StarPU init"); }

  using FR_Tests::SetUp;

  static void TearDownTestSuite() { starpu_shutdown(); }

  PhysicalDescriptor Create(size_t bytes, DataType type) override
  {
    void *ptr = nullptr;
    if (0 != bytes) {
      starpu_malloc(&ptr, bytes);
      memory.emplace_back(ptr);
      memset(ptr, 0, bytes);
    }
    auto *handle = new starpu_data_handle_t();
    return PhysicalDescriptor(ptr, bytes, type, ManagedBy::StarPU, handle);
  }

  void Remove(const PhysicalDescriptor &pd) override
  {
    if (auto it = std::find(memory.begin(), memory.end(), pd.GetPtr()); memory.end() != it) {
      starpu_free(pd.GetPtr());
      memory.erase(it);
    }
    delete StarPUUtils::GetHandle(pd);
  }

  void Register(const PhysicalDescriptor &pd) override { StarPUUtils::Register(pd); };

  void Unregister(const PhysicalDescriptor &pd) override
  {
    StarPUUtils::Unregister(pd, StarPUUtils::UnregisterType::kSubmitNoCopy);
  };

  void Acquire(const PhysicalDescriptor &pd) override
  {
    starpu_data_acquire(*StarPUUtils::GetHandle(pd), STARPU_RW);
  }

  void Release(const PhysicalDescriptor &pd) override
  {
    starpu_data_release(*StarPUUtils::GetHandle(pd));
  }

private:
  FRStarPU transformer;
  std::vector<void *> memory;
};

#define NAME FRStarPUTest
#include <tests/operations/fourier_reconstruction/fr_tests.hpp>