#pragma once

#include <libumpalumpa/algorithms/initialization/abstract.hpp>

namespace umpalumpa::initialization {
class CPU : public Abstract
{
public:
  using BasicAlgorithm::Strategy;
  void Synchronize() override{};

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::initialization
