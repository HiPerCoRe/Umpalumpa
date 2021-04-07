#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>

namespace umpalumpa {
namespace extrema_finder {
  class SingleExtremaFinderStarPU : public AExtremaFinder
  {
  public:
    bool Execute(const ResultData &out, const SearchData &in, const Settings &settings) override;

    private:
    inline static const std::string taskName = "Single Extrema Finder";
  };
}// namespace extrema_finder
}// namespace umpalumpa
