#include <libumpalumpa/tuning/ktt_provider.hpp>

namespace umpalumpa {
namespace utils {

  KTTHelper &KTTProvider::Get(int workerId) { return *KTTProvider::Get().map.at(workerId); }

  KTTProvider &KTTProvider::Get()
  {
    static const std::unique_ptr<KTTProvider> instance{ new KTTProvider() };
    return *instance;
  }

  void KTTProvider::Ensure(int workerId, const std::vector<ktt::ComputeQueue> &queues)
  {
    assert(queues.size() >= 1);
    std::lock_guard<std::mutex> lck(KTTProvider::Get().mutex);
    auto &provider = KTTProvider::Get();
    auto it = provider.map.find(workerId);
    if (provider.map.end() == it) {
      // not found, insert it
      // Use first queue to determine the CUDA context
      provider.map[workerId] = std::make_unique<KTTHelper>(
        provider.GetContext(static_cast<CUstream>(queues.at(0))), queues);
    }
  }

  CUcontext KTTProvider::GetContext(CUstream stream) const
  {
    CudaErrchk(cuInit(0));
    CUcontext context;
    CudaErrchk(cuStreamGetCtx(stream, &context));
    return context;
  }

}// namespace utils
}// namespace umpalumpa
