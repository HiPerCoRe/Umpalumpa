#pragma once
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>

namespace umpalumpa {
namespace fourier_transformation {
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
}// namespace fourier_transformation
}// namespace umpalumpa
