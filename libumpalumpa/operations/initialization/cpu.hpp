#pragma once

#include <libumpalumpa/operations/initialization/abstract.hpp>

namespace umpalumpa::initialization {
class CPU : public Abstract
{
public:
  using BasicOperation::Strategy;
  void Synchronize() override{};

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::initialization
