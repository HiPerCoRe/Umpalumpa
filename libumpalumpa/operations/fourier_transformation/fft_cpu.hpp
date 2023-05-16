#pragma once
#include <libumpalumpa/operations/fourier_transformation/afft.hpp>

namespace umpalumpa::fourier_transformation {
class FFTCPU final : public AFFT
{
public:
  using BasicOperation::Strategy;

  void Synchronize() override{
    // nothing to do
  };

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::fourier_transformation
