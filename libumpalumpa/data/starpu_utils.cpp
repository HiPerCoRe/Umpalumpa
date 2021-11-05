#include <libumpalumpa/data/starpu_utils.hpp>

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
