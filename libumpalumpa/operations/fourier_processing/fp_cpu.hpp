#pragma once

#include <libumpalumpa/operations/fourier_processing/afp.hpp>

namespace umpalumpa::fourier_processing {
class FPCPU : public AFP
{
public:
  void Synchronize() override {}
  using BasicOperation::Strategy;

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::fourier_processing
