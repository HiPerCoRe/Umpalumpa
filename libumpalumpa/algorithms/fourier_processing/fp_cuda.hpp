#pragma once

#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>

namespace umpalumpa::fourier_processing {
class FPCUDA
  : public AFP
  , public algorithm::KTT_Base
{
public:
  using algorithm::KTT_Base::KTT_Base;
  using BasicAlgorithm::Strategy;
  void Synchronize() override;

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};
}// namespace umpalumpa::fourier_processing
