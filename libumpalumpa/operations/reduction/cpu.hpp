#pragma once

#include <libumpalumpa/operations/reduction/abstract.hpp>

namespace umpalumpa::reduction {
class CPU : public Abstract
{
public:
  using BasicOperation::Strategy;
  void Synchronize() override{};

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::reduction
