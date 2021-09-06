#pragma once

#include <libumpalumpa/system_includes/ktt.hpp>

typedef struct CUstream_st *CUstream;

namespace umpalumpa {
namespace algorithm {
  class KTT_Base
  {
  public:
    explicit KTT_Base(CUstream stream)
      : tuner(ktt::ComputeApi::CUDA, createApiInitializer(stream)){};
    explicit KTT_Base(int deviceOrdinal)
      : tuner(ktt::ComputeApi::CUDA, createApiInitializer(deviceOrdinal)){};

  protected:
    ktt::ComputeApiInitializer createApiInitializer(int deviceOrdinal);
    ktt::ComputeApiInitializer createApiInitializer(CUstream stream);
    ktt::Tuner tuner;
  };
}// namespace algorithm 
}// namespace umpalumpa


