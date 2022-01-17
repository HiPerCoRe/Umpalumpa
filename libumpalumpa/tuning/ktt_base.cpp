#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/utils/cuda.hpp>

namespace umpalumpa::tuning {

void KTT_Base::CreateStreams()
{
  CudaErrchk(cuInit(0));
  CUdevice device;
  // assuming KTTId == workerId == deviceOrdinal (otherwise user should provide streams)
  CudaErrchk(cuDeviceGet(&device, KTTId));
  CUcontext context;
  cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
  for (unsigned i = 0; i < GetNoOfStreams(); ++i) {
    CUstream s;
    CudaErrchk(cuStreamCreate(&s, CU_STREAM_DEFAULT));
    streams.emplace_back(s);
  }
}

std::vector<ktt::ComputeQueue> KTT_Base::CreateComputeQueues() const
{
  std::vector<ktt::ComputeQueue> result;
  result.reserve(streams.size());
  std::copy(streams.begin(), streams.end(), std::back_inserter(result));
  return result;
}

}// namespace umpalumpa::tuning
