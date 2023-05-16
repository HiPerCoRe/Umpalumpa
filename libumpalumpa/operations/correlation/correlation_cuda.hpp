#pragma once

#include <libumpalumpa/operations/correlation/acorrelation.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/ktt_strategy_base.hpp>

namespace umpalumpa::correlation {
class Correlation_CUDA
  : public ACorrelation
  , public tuning::KTT_Base
{
public:
  using tuning::KTT_Base::KTT_Base;
  using BasicOperation::Strategy;
  using KTTStrategy = tuning::KTTStrategyBase<OutputData, InputData, Settings>;
  void Synchronize() override;

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::correlation
