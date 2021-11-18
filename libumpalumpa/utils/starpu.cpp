#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::utils {

std::vector<unsigned> StarPUUtils::GetCPUWorkerIDs(unsigned n)
{
  const auto cpuWorkerCount = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
  if (cpuWorkerCount < 1) {
    spdlog::error("[StarPU] Return wrong number of CPU workers ({})", cpuWorkerCount);
  }
  auto ids = std::make_unique<int[]>(static_cast<size_t>(cpuWorkerCount));

  auto count = starpu_worker_get_ids_by_type(
    STARPU_CPU_WORKER, ids.get(), static_cast<unsigned>(cpuWorkerCount));

  auto mask = std::vector<unsigned>();
  for (auto i = 0u; i < count; i += n) { mask.emplace_back(ids[i]); }
  return mask;
}

void StarPUUtils::Register(const data::PhysicalDescriptor &pd)
{
  auto nx = (0 == pd.GetBytes()) ? 0 : static_cast<uint32_t>(pd.GetBytes() / Sizeof(pd.GetType()));
  spdlog::debug("[StarPU] Registering {} bytes at {} with handle {}",
    pd.GetBytes(),
    fmt::ptr(pd.GetPtr()),
    fmt::ptr(pd.GetHandle()));
  starpu_vector_data_register(reinterpret_cast<starpu_data_handle_t *>(pd.GetHandle()),
    STARPU_MAIN_RAM,
    reinterpret_cast<uintptr_t>(pd.GetPtr()),
    nx,
    Sizeof(pd.GetType()));
}

void *StarPUUtils::ReceivePDPtr(void *buffer)
{
  return reinterpret_cast<void *>(STARPU_VECTOR_GET_PTR(buffer));
}

void StarPUUtils::Unregister(const data::PhysicalDescriptor &pd)
{
  spdlog::debug("[StarPU] Unregistering handle {}", fmt::ptr(pd.GetHandle()));
  starpu_data_unregister(*reinterpret_cast<starpu_data_handle_t *>(pd.GetHandle()));
}

}// namespace umpalumpa::utils
