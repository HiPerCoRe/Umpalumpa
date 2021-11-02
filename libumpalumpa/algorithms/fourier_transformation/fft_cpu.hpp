#pragma once
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>

namespace umpalumpa::fourier_transformation {
class FFTCPU final : public AFFT
{
public:
  using BasicAlgorithm::Strategy;

  void Synchronize() override{
    // nothing to do
  };

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::fourier_transformation
