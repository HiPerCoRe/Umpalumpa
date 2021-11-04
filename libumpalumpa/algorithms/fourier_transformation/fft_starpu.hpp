#pragma once
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/data/starpu_payload.hpp>
#include <vector>
#include <memory>

namespace umpalumpa::fourier_transformation {
class FFTStarPU final : public AFFT
{

public:
  void Synchronize() override{
    // don't do anything. Each task is synchronized, now it's StarPU's problem
    // consider calling starpu_task_wait_for_all() instead
  };

  // make 'normal' Init() and Execute() available
  using AFFT::Init;
  using AFFT::Execute;

  using StarpuOutputData =
    OutputDataWrapper<std::unique_ptr<data::StarpuPayload<OutputData::PayloadType::LDType>>>;
  using StarpuInputData =
    InputDataWrapper<std::unique_ptr<data::StarpuPayload<OutputData::PayloadType::LDType>>>;

  [[nodiscard]] bool
    Init(const StarpuOutputData &out, const StarpuInputData &in, const Settings &s);
  [[nodiscard]] bool Execute(const StarpuOutputData &out, const StarpuInputData &in);

protected:
  bool InitImpl() override;
  bool ExecuteImpl(const OutputData &out, const InputData &in);
  bool ExecuteImpl(const StarpuOutputData &out, const StarpuInputData &in);

private:
  inline static const std::string taskName = "FFT StarPU";

  std::vector<std::unique_ptr<AFFT>> algs;
  const StarpuOutputData *outPtr = nullptr;
  const StarpuInputData *inPtr = nullptr;
};
}// namespace umpalumpa::fourier_transformation
