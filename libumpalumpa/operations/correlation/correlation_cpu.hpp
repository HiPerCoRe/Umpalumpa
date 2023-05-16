#pragma once

#include <libumpalumpa/operations/correlation/acorrelation.hpp>

namespace umpalumpa::correlation {
class Correlation_CPU : public ACorrelation
{
public:
  using BasicOperation::Strategy;
  void Synchronize() override{};

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::correlation
