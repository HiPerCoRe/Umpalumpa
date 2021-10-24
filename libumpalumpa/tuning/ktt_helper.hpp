#pragma once

#include <mutex>
#include <libumpalumpa/system_includes/ktt.hpp>
#include <libumpalumpa/utils/cuda.hpp>
#include <libumpalumpa/utils/system.hpp>

namespace umpalumpa {
namespace utils {
  class KTTHelper
  {

  public:
    KTTHelper(const CUcontext context, const std::vector<ktt::ComputeQueue> &queues)
      : tuner(ktt::ComputeApi::CUDA, ktt::ComputeApiInitializer(context, queues))
    {
      tuner.SetCompilerOptions("-I" + kProjectRoot + " --std=c++14 -default-device");
    }

    std::mutex &GetMutex() { return mutex; }

    ktt::Tuner &GetTuner() { return tuner; }

    void AddQueue(const ktt::ComputeQueue)
    {
      // FIXME implement once KTT supports it
      // ensure that you don't register twice the same queue
    }

    void RemoveQueue(const ktt::ComputeQueue)
    {
      // FIXME implment once KTT supports it
      // be careful not to remove queue used by another algorithm
    }

    // FIXME add method to access correct stream (and remove interface.GetAllQueues().at(0) from
    // strategy calls)

  private:
    ktt::Tuner tuner;
    std::mutex mutex;
  };

}// namespace utils
}// namespace umpalumpa
