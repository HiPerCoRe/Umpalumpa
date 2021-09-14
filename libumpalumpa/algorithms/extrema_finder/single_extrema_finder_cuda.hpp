#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <libumpalumpa/algorithms/ktt_base.hpp>
#include <vector>
#include <memory>
#include <map>

namespace umpalumpa {
namespace extrema_finder {
  class SingleExtremaFinderCUDA : public AExtremaFinder, public algorithm::KTT_Base
  {
  public:
    using algorithm::KTT_Base::KTT_Base;

    bool Init(const ResultData &out, const SearchData &in, const Settings &settings) override;
    bool Execute(const ResultData &out, const SearchData &in, const Settings &settings) override;

    struct Strategy
    {
      virtual ~Strategy() = default;
      virtual bool
        Init(const ResultData &, const SearchData &, const Settings &s, ktt::Tuner &tuner) = 0;
      virtual bool Execute(const ResultData &out,
        const SearchData &in,
        const Settings &settings,
        ktt::Tuner &tuner) = 0;
      virtual std::string GetName() const = 0;

      struct KernelData
      {
        ktt::KernelDefinitionId definitionId;// FIXME this should be a vector
        ktt::KernelId kernelId;
      };
      // FIXME add std::vector ktt arguments that shall be deteled in.e.g destructor or synchronize() method
    };

    void Synchronize() override;

  private:
    std::unique_ptr<Strategy> strategy;
  };
}// namespace extrema_finder
}// namespace umpalumpa
