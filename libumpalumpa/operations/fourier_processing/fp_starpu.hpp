#pragma once
#include <libumpalumpa/operations/fourier_processing/afp.hpp>
#include <queue>

// forward declaration
struct starpu_task;

namespace umpalumpa::fourier_processing {
class FPStarPU final : public AFP
{
public:
  virtual ~FPStarPU();

  void Cleanup() override;

  void Synchronize() override;

protected:
  bool InitImpl() override;
  bool ExecuteImpl(const OutputData &out, const InputData &in);

private:
  inline static const std::string taskName = "FourierProcessingStarPU";

  std::vector<AFP *> ops;
  long noOfInitWorkers = 0;

  std::queue<starpu_task *> taskQueue;
};
}// namespace umpalumpa::fourier_processing
