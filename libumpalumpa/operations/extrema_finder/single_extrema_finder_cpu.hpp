#pragma once

#include <libumpalumpa/operations/extrema_finder/aextrema_finder.hpp>

namespace umpalumpa::extrema_finder {
class SingleExtremaFinderCPU : public AExtremaFinder
{
public:
  using BasicOperation::Strategy;
  void Synchronize() override{};

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::extrema_finder
