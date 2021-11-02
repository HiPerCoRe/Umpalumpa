#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>

namespace umpalumpa::extrema_finder {
class SingleExtremaFinderCPU : public AExtremaFinder
{
public:
  using BasicAlgorithm::Strategy;
  void Synchronize() override{};

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::extrema_finder
