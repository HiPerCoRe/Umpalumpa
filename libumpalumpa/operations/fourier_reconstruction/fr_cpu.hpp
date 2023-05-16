#pragma once

#include <libumpalumpa/operations/fourier_reconstruction/afr.hpp>

namespace umpalumpa::fourier_reconstruction {
class FRCPU : public AFR
{
public:
  void Synchronize() override {}
  using BasicOperation::Strategy;

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};

}// namespace umpalumpa::fourier_reconstruction