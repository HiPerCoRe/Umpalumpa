#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>

namespace umpalumpa {
namespace extrema_finder {
  class SingleExtremaFinderGPU : public AExtremaFinder
  {
  public:
    bool Execute(const ResultData &out, const SearchData &in, const Settings &settings) override;
  };
}// namespace extrema_finder
}// namespace umpalumpa
