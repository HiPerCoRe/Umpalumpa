#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/afr.hpp>

namespace umpalumpa::fourier_reconstruction {
class FRCPU : public AFR
{
public:
  void Synchronize() override {}
  using BasicAlgorithm::Strategy;

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};

}// namespace umpalumpa::fourier_reconstruction