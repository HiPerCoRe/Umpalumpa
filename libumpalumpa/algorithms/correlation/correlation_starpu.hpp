#pragma once
#include <libumpalumpa/algorithms/correlation/acorrelation.hpp>
#include <queue>

// forward declaration
struct starpu_task;

namespace umpalumpa::correlation {
class Correlation_StarPU final : public ACorrelation
{
public:
  ~Correlation_StarPU();

  void Cleanup() override;

  void Synchronize() override;

protected:
  bool InitImpl() override;
  bool ExecuteImpl(const OutputData &out, const InputData &in);

private:
  inline static const std::string taskName = "CorrelationStarPU";

  /**
   * Holds pointers to used algorithms.
   * Notice that those pointer (if any) refer to the worker-specific
   * memory nodes, i.e. it is safe to access them only on 'their' workers.
   * Only nonnull algorithms are properly initialized.
   **/
  std::vector<ACorrelation *> algs;
  long noOfInitWorkers = 0;

  std::queue<starpu_task *> taskQueue;
};
}// namespace umpalumpa::correlation
