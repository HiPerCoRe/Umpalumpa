#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>

namespace umpalumpa::extrema_finder {
class SingleExtremaFinderCUDA
  : public AExtremaFinder
  , public algorithm::KTT_Base
{
public:
  using algorithm::KTT_Base::KTT_Base;
  using BasicAlgorithm::Strategy;
  void Synchronize() override;

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::extrema_finder
