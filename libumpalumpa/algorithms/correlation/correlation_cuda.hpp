#pragma once

#include <libumpalumpa/algorithms/correlation/acorrelation.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <vector>
#include <memory>
#include <map>

namespace umpalumpa::correlation {
class Correlation_CUDA
  : public ACorrelation
  , public algorithm::KTT_Base
{
public:
  using algorithm::KTT_Base::KTT_Base;
  using BasicAlgorithm::Strategy;
  void Synchronize() override;

  const Correlation_CUDA &Get() const override { return *this; }

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::correlation
