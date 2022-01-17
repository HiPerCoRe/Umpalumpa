#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/ktt_strategy_base.hpp>

namespace umpalumpa::extrema_finder {
class SingleExtremaFinderCUDA
  : public AExtremaFinder
  , public tuning::KTT_Base
{
public:
  using tuning::KTT_Base::KTT_Base;
  using BasicAlgorithm::Strategy;
  using KTTStrategy = tuning::KTTStrategyBase<OutputData, InputData, Settings>;
  void Synchronize() override;

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::extrema_finder
