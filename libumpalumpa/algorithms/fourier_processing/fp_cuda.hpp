#pragma once

#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>
#include <libumpalumpa/system_includes/ktt.hpp>
#include <vector>
#include <memory>
#include <map>

typedef struct CUstream_st *CUstream;

namespace umpalumpa {
namespace fourier_processing {
  class FP_CUDA : public AFP
  {
  public:
    explicit FP_CUDA(CUstream stream)
      : tuner(ktt::ComputeApi::CUDA, createApiInitializer(stream)){};
    explicit FP_CUDA(int deviceOrdinal)
      : tuner(ktt::ComputeApi::CUDA, createApiInitializer(deviceOrdinal)){};

    virtual bool Init(const OutputData &out, const InputData &in, const Settings &settings) = 0;
    virtual bool Execute(const OutputData &out, const InputData &in) = 0;

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

    void Synchronize();

  private:
    ktt::ComputeApiInitializer createApiInitializer(int deviceOrdinal);
    ktt::ComputeApiInitializer createApiInitializer(CUstream stream);
    std::unique_ptr<Strategy> strategy;
    ktt::Tuner tuner;
  };
}// namespace fourier_processing 
}// namespace umpalumpa

