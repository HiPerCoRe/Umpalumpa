#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <vector>
#include <memory>
#include <map>

namespace umpalumpa {
namespace extrema_finder {
  class SingleExtremaFinderCUDA
    : public AExtremaFinder
    , public algorithm::KTT_Base
  {
  public:
    using algorithm::KTT_Base::KTT_Base;

    bool Init(const OutputData &out, const InputData &in, const Settings &settings) override;
    bool Execute(const OutputData &out, const InputData &in, const Settings &settings) override;

    struct Strategy : public algorithm::TunableStrategy
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
      // FIXME add std::vector ktt arguments that shall be deteled in.e.g destructor or
      // synchronize() method
    };

    void Synchronize() override;

  private:
    std::unique_ptr<Strategy> strategy;
  };
}// namespace extrema_finder
}// namespace umpalumpa
