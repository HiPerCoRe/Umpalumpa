#pragma once
#include <libumpalumpa/algorithms/correlation/acorrelation.hpp>
#include <libumpalumpa/data/starpu_payload.hpp>

namespace umpalumpa::correlation {
class CorrelationStarPU final : public ACorrelation
{

public:
  void Synchronize() override{
    // don't do anything. Each task is synchronized, now it's StarPU's problem
    // consider calling starpu_task_wait_for_all() instead
  };

  using StarpuOutputData =
    OutputDataWrapper<std::unique_ptr<data::StarpuPayload<data::FourierDescriptor>>>;
  using StarpuInputData =
    InputDataWrapper<std::unique_ptr<data::StarpuPayload<data::FourierDescriptor>>>;

  [[nodiscard]] bool
    Init(const StarpuOutputData &out, const StarpuInputData &in, const Settings &s);
  [[nodiscard]] bool Execute(const StarpuOutputData &out, const StarpuInputData &in);

protected:
  bool InitImpl() override;
  bool ExecuteImpl(const OutputData &out, const InputData &in);
  bool ExecuteImpl(const StarpuOutputData &out, const StarpuInputData &in);

private:
  inline static const std::string taskName = "Correlation StarPU";

  std::vector<std::unique_ptr<ACorrelation>> algs;
  const StarpuOutputData *outPtr = nullptr;
  const StarpuInputData *inPtr = nullptr;
};
}// namespace umpalumpa::correlation
