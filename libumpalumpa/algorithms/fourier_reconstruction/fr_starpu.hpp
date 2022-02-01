#pragma once
#include <libumpalumpa/algorithms/fourier_reconstruction/afr.hpp>
#include <queue>

// forward declaration
struct starpu_task;

namespace umpalumpa::fourier_reconstruction {
class FRStarPU final : public AFR
{
public:
  virtual ~FRStarPU();

  void Cleanup() override;

  void Synchronize() override;

protected:
  bool InitImpl() override;
  bool ExecuteImpl(const OutputData &out, const InputData &in);

private:
  inline static const std::string taskName = "FourierReconstructionStarPU";

  std::vector<AFR *> algs;
  long noOfInitWorkers = 0;

  std::queue<starpu_task *> taskQueue;
};
}// namespace umpalumpa::fourier_reconstruction
