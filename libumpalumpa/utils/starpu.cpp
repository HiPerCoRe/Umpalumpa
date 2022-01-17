#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::utils {

std::vector<unsigned> StarPUUtils::GetCPUWorkerIDs(unsigned n)
{
  const auto cpuWorkerCount = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
  if (cpuWorkerCount < 1) {
    auto level = (0 == cpuWorkerCount) ? spdlog::level::warn : spdlog::level::err;
    spdlog::log(level, "[StarPU] Wrong number of CPU workers detected ({})", cpuWorkerCount);
  }
  auto ids = std::make_unique<int[]>(static_cast<size_t>(cpuWorkerCount));

  auto count = starpu_worker_get_ids_by_type(
    STARPU_CPU_WORKER, ids.get(), static_cast<unsigned>(cpuWorkerCount));

  auto mask = std::vector<unsigned>();
  for (auto i = 0u; i < count; i += n) { mask.emplace_back(ids[i]); }
  return mask;
}

void StarPUUtils::Register(const data::PhysicalDescriptor &pd, int home_node)
{
  spdlog::debug("[StarPU] Registering {} bytes at {} with handle {}",
    pd.GetBytes(),
    fmt::ptr(pd.GetPtr()),
    fmt::ptr(pd.GetHandle()));
  if (0 == pd.GetBytes()) { return starpu_void_data_register(GetHandle(pd)); }
  starpu_vector_data_register(GetHandle(pd),
    home_node,
    reinterpret_cast<uintptr_t>(pd.GetPtr()),
    static_cast<uint32_t>(pd.GetBytes() / Sizeof(pd.GetType())),
    Sizeof(pd.GetType()));
}

void *StarPUUtils::ReceivePDPtr(void *buffer)
{
  return reinterpret_cast<void *>(STARPU_VECTOR_GET_PTR(buffer));
}

void StarPUUtils::Unregister(const data::PhysicalDescriptor &pd, UnregisterType type)
{
  spdlog::debug("[StarPU] Unregistering handle {}", fmt::ptr(pd.GetHandle()));
  switch (type) {
  case UnregisterType::kBlockingCopyToHomeNode:
    return starpu_data_unregister(*reinterpret_cast<starpu_data_handle_t *>(pd.GetHandle()));
  case UnregisterType::kBlockingNoCopy:
    return starpu_data_unregister_no_coherency(
      *reinterpret_cast<starpu_data_handle_t *>(pd.GetHandle()));
  case UnregisterType::kSubmitNoCopy:
    return starpu_data_unregister_submit(*reinterpret_cast<starpu_data_handle_t *>(pd.GetHandle()));
  default:
    spdlog::error("Unsupported Unregister type");
  }
}

}// namespace umpalumpa::utils
