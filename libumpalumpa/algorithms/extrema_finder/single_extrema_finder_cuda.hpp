#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <libumpalumpa/system_includes/ktt.hpp>
#include <vector>
#include <memory>
#include <map>

typedef struct CUstream_st *CUstream;

namespace umpalumpa {
namespace extrema_finder {
  class SingleExtremaFinderCUDA : public AExtremaFinder
  {
  public:
    explicit SingleExtremaFinderCUDA(int deviceOrdinal, CUstream stream)
      : tuner(ktt::ComputeApi::CUDA, createApiInitializer(deviceOrdinal, stream)){};
    explicit SingleExtremaFinderCUDA(int deviceOrdinal)
      : tuner(ktt::ComputeApi::CUDA, createApiInitializer(deviceOrdinal)){};
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
    };

    void Synchronize();

  private:
    ktt::ComputeApiInitializer createApiInitializer(int deviceOrdinal);
    ktt::ComputeApiInitializer createApiInitializer(int deviceOrdinal, CUstream stream);
    std::unique_ptr<Strategy> strategy;
    ktt::Tuner tuner;
  };
}// namespace extrema_finder
}// namespace umpalumpa
