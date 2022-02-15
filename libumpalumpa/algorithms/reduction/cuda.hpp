#pragma once

#include <libumpalumpa/algorithms/reduction/abstract.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/ktt_strategy_base.hpp>

namespace umpalumpa::reduction {
class CUDA
  : public Abstract
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
}// namespace umpalumpa::reduction
