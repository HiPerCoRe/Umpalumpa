#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <libumpalumpa/data/starpu_payload.hpp>

namespace umpalumpa::extrema_finder {
class SingleExtremaFinderStarPU : public AExtremaFinder
{
public:
  void Synchronize() override{
    // don't do anything. Each task is synchronized, now it's StarPU's problem
    // consider calling starpu_task_wait_for_all() instead
  };

  using StarpuOutputData =
    OutputDataWrapper<std::unique_ptr<data::StarpuPayload<data::LogicalDescriptor>>>;
  using StarpuInputData =
    InputDataWrapper<std::unique_ptr<data::StarpuPayload<data::LogicalDescriptor>>>;

  [[nodiscard]] bool
    Init(const StarpuOutputData &out, const StarpuInputData &in, const Settings &s);
  [[nodiscard]] bool Execute(const StarpuOutputData &out, const StarpuInputData &in);

protected:
  bool InitImpl() override;
  bool ExecuteImpl(const OutputData &out, const InputData &in);
  bool ExecuteImpl(const StarpuOutputData &out, const StarpuInputData &in);

private:
  inline static const std::string taskName = "Single Extrema Finder StarPU";

  std::vector<std::unique_ptr<AExtremaFinder>> algs;
  const StarpuOutputData *outPtr = nullptr;
  const StarpuInputData *inPtr = nullptr;
};
}// namespace umpalumpa::extrema_finder
