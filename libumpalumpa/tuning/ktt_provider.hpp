#pragma once

#include <memory>
#include <mutex>
#include <map>
#include <cassert>
#include <libumpalumpa/tuning/ktt_helper.hpp>

namespace umpalumpa {
namespace utils {

  class KTTProvider
  {
  public:
    static void Ensure(int workerId, const std::vector<ktt::ComputeQueue> &queues);
    static KTTHelper &Get(int workerId);

  protected:
    KTTProvider(const KTTProvider &) = delete;
    KTTProvider &operator=(const KTTProvider) = delete;
    KTTProvider(){};

    static KTTProvider &Get();

  private:
    CUcontext GetContext(CUstream stream) const;

    std::mutex mutex;
    std::map<int, std::unique_ptr<KTTHelper>> map;
  };

}// namespace utils
}// namespace umpalumpa
