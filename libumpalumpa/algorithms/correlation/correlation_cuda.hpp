#pragma once

#include <libumpalumpa/algorithms/correlation/acorrelation.hpp>
#include <libumpalumpa/system_includes/ktt.hpp>
#include <vector>
#include <memory>
#include <map>

typedef struct CUstream_st *CUstream;

namespace umpalumpa {
namespace correlation {
  class Correlation_CUDA : public ACorrelation 
  {
  public:
    explicit Correlation_CUDA(CUstream stream)
      : tuner(ktt::ComputeApi::CUDA, createApiInitializer(stream)){};
    explicit Correlation_CUDA(int deviceOrdinal)
      : tuner(ktt::ComputeApi::CUDA, createApiInitializer(deviceOrdinal)){};

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
    ktt::ComputeApiInitializer createApiInitializer(int deviceOrdinal);
    ktt::ComputeApiInitializer createApiInitializer(CUstream stream);
    std::unique_ptr<Strategy> strategy;
    ktt::Tuner tuner;
  };
}// namespace correlation 
}// namespace umpalumpa

