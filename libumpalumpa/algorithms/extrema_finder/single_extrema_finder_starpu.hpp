#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <libumpalumpa/data/starpu_payload.hpp>
#include <vector>
#include <memory>

namespace umpalumpa {
namespace extrema_finder {
  class SingleExtremaFinderStarPU : public AExtremaFinder
  {
  public:
    bool Init(const ResultData &out, const SearchData &in, const Settings &settings) override;
    bool Execute(const ResultData &out, const SearchData &in, const Settings &settings) override;
    void Synchronize(){
      // don't do anything. Each task is synchronized, now it's StarPU's problem
      // consider calling starpu_task_wait_for_all() instead
    };

    using StarpuResultData =
      ResultDataWrapper<std::unique_ptr<data::StarpuPayload<ResultData::type::type>>>;

    using StarpuSearchData =
      SearchDataWrapper<std::unique_ptr<data::StarpuPayload<SearchData::type::type>>>;

    bool Execute(const StarpuResultData &out, const StarpuSearchData &in, const Settings &settings);

  private:
    inline static const std::string taskName = "Single Extrema Finder";

    std::vector<std::unique_ptr<AExtremaFinder>> algs;
  };
}// namespace extrema_finder
}// namespace umpalumpa
