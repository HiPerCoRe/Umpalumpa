#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <libumpalumpa/utils/ktt.hpp>
#include <vector>
#include <memory>
#include <map>

namespace umpalumpa {
namespace extrema_finder {
  class SingleExtremaFinderCUDA : public AExtremaFinder
  {
  public:
    SingleExtremaFinderCUDA(ktt::DeviceIndex deviceIndex)
      : tuner(0, deviceIndex, ktt::ComputeApi::CUDA){};
    bool Init(const ResultData &out, const SearchData &in, const Settings &settings) override;
    bool Execute(const ResultData &out, const SearchData &in, const Settings &settings) override;

    struct Strategy
    {
      virtual ~Strategy() = default;
      virtual bool Init(const ResultData &, const SearchData &, const Settings &s, ktt::Tuner &tuner) = 0;
      virtual bool Execute(const ResultData &out,
        const SearchData &in,
        const Settings &settings,
        ktt::Tuner &tuner) = 0;
      virtual std::string GetName() const = 0;

      struct KernelData
      {
        ktt::KernelDefinitionId definitionId; ///this should be vector
        ktt::KernelId kernelId;
      };
    };

  private:
    std::unique_ptr<Strategy> strategy;
    ktt::Tuner tuner;
  };
}// namespace extrema_finder
}// namespace umpalumpa
