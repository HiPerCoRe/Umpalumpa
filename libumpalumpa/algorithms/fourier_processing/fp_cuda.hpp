#pragma once

#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>
#include <libumpalumpa/algorithms/ktt_base.hpp>
#include <vector>
#include <memory>
#include <map>

namespace umpalumpa {
namespace fourier_processing {
  class FP_CUDA
    : public AFP
    , public algorithm::KTT_Base
  {
  public:
    using algorithm::KTT_Base::KTT_Base;

    virtual bool
      Init(const OutputData &out, const InputData &in, const Settings &settings) override;
    virtual bool Execute(const OutputData &out, const InputData &in) override;

    struct Strategy
    {
      virtual ~Strategy() = default;
      virtual bool Init(const OutputData &,
        const InputData &,
        const Settings &s,
        utils::KTTHelper &helper) = 0;
      virtual bool Execute(const OutputData &out,
        const InputData &in,
        const Settings &settings,
        utils::KTTHelper &helper) = 0;
      virtual std::string GetName() const = 0;
      std::string GetFullName() const { return typeid(*this).name(); }
    };

    void Synchronize() override;

  private:
    std::unique_ptr<Strategy> strategy;
  };
}// namespace fourier_processing
}// namespace umpalumpa
