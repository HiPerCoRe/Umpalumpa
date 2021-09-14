#pragma once

#include <libumpalumpa/algorithms/correlation/acorrelation.hpp>
#include <libumpalumpa/algorithms/ktt_base.hpp>
#include <vector>
#include <memory>
#include <map>

namespace umpalumpa {
namespace correlation {
  class Correlation_CUDA : public ACorrelation, public algorithm::KTT_Base
  {
  public:
    using algorithm::KTT_Base::KTT_Base;

    virtual bool Init(const OutputData &out, const InputData &in, const Settings &settings) override;
    virtual bool Execute(const OutputData &out, const InputData &in) override;

    struct Strategy
    {
      virtual ~Strategy() = default;
      virtual bool
        Init(const OutputData &, const InputData &, const Settings &s, ktt::Tuner &tuner) = 0;
      virtual bool Execute(const OutputData &out,
        const InputData &in,
        const Settings &settings,
        ktt::Tuner &tuner) = 0;
      virtual std::string GetName() const = 0;

      struct KernelData
      {
        std::vector<ktt::KernelDefinitionId> definitionIds;
        ktt::KernelId kernelId;
        std::vector<ktt::ArgumentId> arguments;
      };
    };

    void Synchronize() override;

  private:
    std::unique_ptr<Strategy> strategy;
  };
}// namespace correlation 
}// namespace umpalumpa

