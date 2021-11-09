#include <libumpalumpa/data/starpu_utils.hpp>
#include <libumpalumpa/utils/cuda.hpp>

std::vector<unsigned> GetCPUWorkerIDs(unsigned n)
{
  const auto cpuWorkerCount = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
  auto ids = std::make_unique<int[]>(static_cast<size_t>(cpuWorkerCount));

  auto count = starpu_worker_get_ids_by_type(
    STARPU_CPU_WORKER, ids.get(), static_cast<unsigned>(cpuWorkerCount));

  auto mask = std::vector<unsigned>(count / n + 1);
  for (auto i = 0ul; i < mask.size(); ++i) { mask[i] = static_cast<unsigned>(ids[i * n]); }
  return mask;
}


unsigned GetMemoryNode(const umpalumpa::data::PhysicalDescriptor &pd)
{
  using umpalumpa::data::ManagedBy;
  switch (pd.GetManager()) {
  case ManagedBy::StarPU:
    return static_cast<unsigned>(pd.GetMemoryNode());
  case ManagedBy::CUDA:
    return STARPU_CPU_RAM;// FIXME can we find out if managed memory is on GPU already?
  case ManagedBy::Manually:
    return (IsGpuPointer(pd.GetPtr()) ? STARPU_CUDA_RAM : STARPU_CPU_RAM);
  default:
    break;
  }
  return STARPU_MAIN_RAM;// because we don't know any better
}