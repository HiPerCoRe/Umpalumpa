#pragma once
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/data/starpu_payload.hpp>
#include <vector>
#include <memory>

namespace umpalumpa {
namespace fourier_transformation {
  class FFTStarPU : public AFFT
  {
  public:
    bool Init(const OutputData &out, const InputData &in, const Settings &settings) override;
    bool Execute(const OutputData &out, const InputData &in) override;
    void Synchronize() override{
      // don't do anything. Each task is synchronized, now it's StarPU's problem
      // consider calling starpu_task_wait_for_all() instead
    };

    using StarpuOutputData =
      DataWrapper<std::unique_ptr<data::StarpuPayload<OutputData::PayloadType::LDType>>>;
    using StarpuInputData =
      DataWrapper<std::unique_ptr<data::StarpuPayload<OutputData::PayloadType::LDType>>>;
    bool Execute(const StarpuOutputData &out, const StarpuInputData &in);

  private:
    inline static const std::string taskName = "Single Extrema Finder";

    std::vector<std::unique_ptr<AFFT>> algs;
  };
}// namespace fourier_transformation
}// namespace umpalumpa
