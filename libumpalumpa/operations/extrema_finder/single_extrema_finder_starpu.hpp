#pragma once

#include <libumpalumpa/operations/extrema_finder/aextrema_finder.hpp>
#include <queue>

// forward declaration
struct starpu_task;

namespace umpalumpa::extrema_finder {
class SingleExtremaFinderStarPU : public AExtremaFinder
{
public:
  ~SingleExtremaFinderStarPU();

  void Cleanup() override;

  void Synchronize() override;

protected:
  bool InitImpl() override;
  bool ExecuteImpl(const OutputData &out, const InputData &in);

private:
  inline static const std::string taskName = "SingleExtremaFinderStarPU";

  /**
   * Holds pointers to used operations.
   * Notice that those pointer (if any) refer to the worker-specific
   * memory nodes, i.e. it is safe to access them only on 'their' workers.
   * Only nonnull operations are properly initialized.
   **/
  std::vector<AExtremaFinder *> ops;
  long noOfInitWorkers = 0;

  std::queue<starpu_task *> taskQueue;
};
}// namespace umpalumpa::extrema_finder
