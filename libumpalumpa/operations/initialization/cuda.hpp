#pragma once

#include <libumpalumpa/operations/initialization/abstract.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/ktt_strategy_base.hpp>

namespace umpalumpa::initialization {
class CUDA
  : public Abstract
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
}// namespace umpalumpa::initialization
