#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <vector>
#include <memory>

namespace umpalumpa {
namespace extrema_finder {
  class SingleExtremaFinderStarPU : public AExtremaFinder
  {
  public:
    static SingleExtremaFinderStarPU &Instance()
    {
      static SingleExtremaFinderStarPU instance;
      return instance;
    }

    SingleExtremaFinderStarPU(SingleExtremaFinderStarPU const &) = delete;
    SingleExtremaFinderStarPU(SingleExtremaFinderStarPU &&) = delete;
    SingleExtremaFinderStarPU &operator=(SingleExtremaFinderStarPU const &) = delete;
    SingleExtremaFinderStarPU &operator=(SingleExtremaFinderStarPU &&) = delete;

    bool Init(const ResultData &out, const SearchData &in, const Settings &settings) override;
    bool Execute(const ResultData &out, const SearchData &in, const Settings &settings) override;
    void Synchronize(){
      // don't do anything. Each task is synchronized, now it's StarPU's problem
      // consider calling starpu_task_wait_for_all() instead
    };

  private:
    SingleExtremaFinderStarPU() = default;
    ~SingleExtremaFinderStarPU() = default;
    inline static const std::string taskName = "Single Extrema Finder";

    std::vector<std::unique_ptr<AExtremaFinder>> algs;
  };
}// namespace extrema_finder
}// namespace umpalumpa
