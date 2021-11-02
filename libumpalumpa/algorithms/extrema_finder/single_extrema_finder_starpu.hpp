#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <libumpalumpa/data/starpu_payload.hpp>
#include <vector>
#include <memory>

namespace umpalumpa::extrema_finder {
class SingleExtremaFinderStarPU : public AExtremaFinder
{
public:
  bool Init(const OutputData &out, const InputData &in, const Settings &settings) override;
  bool Execute(const OutputData &out, const InputData &in, const Settings &settings) override;
  void Synchronize() override{
    // don't do anything. Each task is synchronized, now it's StarPU's problem
    // consider calling starpu_task_wait_for_all() instead
  };


  struct StarpuResultData final
    : ResultDataWrapper<std::unique_ptr<data::StarpuPayload<OutputData::type::LDType>>>
  {
    using ptrType = data::StarpuPayload<OutputData::type::LDType>;
    StarpuResultData(type &&vals, type &&locs) : ResultDataWrapper(std::move(vals), std::move(locs))
    {}

    StarpuResultData(const OutputData &d)
      : StarpuResultData(std::make_unique<ptrType>(d.values),
        std::make_unique<ptrType>(d.locations))
    {}
  };

  using StarpuSearchData =
    SearchDataWrapper<std::unique_ptr<data::StarpuPayload<InputData::type::LDType>>>;

  bool Execute(const StarpuResultData &out, const StarpuSearchData &in, const Settings &settings);

private:
  inline static const std::string taskName = "Single Extrema Finder";

  std::vector<std::unique_ptr<AExtremaFinder>> algs;
};
}// namespace umpalumpa::extrema_finder
