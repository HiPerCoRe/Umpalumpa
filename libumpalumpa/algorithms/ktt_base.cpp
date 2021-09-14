#include <libumpalumpa/algorithms/ktt_base.hpp>
#include <libumpalumpa/utils/cuda.hpp>

namespace umpalumpa {
namespace algorithm {

  ktt::ComputeApiInitializer KTT_Base::createApiInitializer(int deviceOrdinal)
  {
    CudaErrchk(cuInit(0));
    CUdevice device;
    CudaErrchk(cuDeviceGet(&device, deviceOrdinal));
    CUcontext context;
    cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
    CUstream stream;
    CudaErrchk(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    return ktt::ComputeApiInitializer(context, std::vector<ktt::ComputeQueue>{ stream });
  }

  ktt::ComputeApiInitializer KTT_Base::createApiInitializer(CUstream stream)
  {
    CudaErrchk(cuInit(0));
    CUcontext context;
    CudaErrchk(cuStreamGetCtx(stream, &context));
    // Create compute API initializer which specifies context and streams that will be utilized by
    // the tuner.
    return ktt::ComputeApiInitializer(context, std::vector<ktt::ComputeQueue>{ stream });
  }

}// namespace algorithm 
}// namespace umpalumpa
