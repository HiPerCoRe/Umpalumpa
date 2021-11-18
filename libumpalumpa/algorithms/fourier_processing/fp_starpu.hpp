#pragma once
#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>

namespace umpalumpa::fourier_processing {
class FPStarPU final : public AFP
{
public:
  void Synchronize() override{
    // don't do anything. Each task is synchronized, now it's StarPU's problem
    // consider calling starpu_task_wait_for_all() instead
  };

protected:
  bool InitImpl() override;
  bool ExecuteImpl(const OutputData &out, const InputData &in);

private:
  inline static const std::string taskName = "Fourier Processing StarPU";

  std::vector<std::unique_ptr<AFP>> algs;

  long noOfInitWorkers = 0;
};
}// namespace umpalumpa::fourier_processing
