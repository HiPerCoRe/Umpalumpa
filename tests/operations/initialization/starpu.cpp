#include <tests/operations/initialization/common.hpp>
#include <libumpalumpa/operations/initialization/starpu.hpp>
#include <libumpalumpa/utils/starpu.hpp>
#include <algorithm>

using umpalumpa::utils::StarPUUtils;

class StarPUTest : public Initialization_Tests
{
public:
  StarPU &GetOp() override { return op; }

  static void SetUpTestSuite() { STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "StarPU init"); }

  using Initialization_Tests::SetUp;

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
    handles.emplace_back(handle);
    return PhysicalDescriptor(ptr, bytes, type, ManagedBy::StarPU, handle);
  }

  void Remove(const PhysicalDescriptor &pd) override
  {
    auto h = StarPUUtils::GetHandle(pd);
    if (auto it = std::find(handles.begin(), handles.end(), h); handles.end() != it) {
      delete StarPUUtils::GetHandle(pd);
      handles.erase(it);
    }
    if (auto it = std::find(memory.begin(), memory.end(), pd.GetPtr()); memory.end() != it) {
      starpu_free(pd.GetPtr());
      memory.erase(it);
    }
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
  StarPU op;
  std::vector<starpu_data_handle_t *> handles;
  std::vector<void *> memory;
};

#define NAME StarPUTest
#include <tests/operations/initialization/tests.hpp>
