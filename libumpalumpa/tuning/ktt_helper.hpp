#pragma once

#include <memory>
#include <mutex>
#include <map>
#include <vector>
#include <libumpalumpa/system_includes/ktt.hpp>
#include <libumpalumpa/utils/cuda.hpp>
#include <libumpalumpa/utils/system.hpp>


namespace umpalumpa {
namespace utils {
  class KTTHelper
  {

  public:
    struct KernelIds
    {
      std::vector<ktt::KernelDefinitionId> definitionIds;
      std::vector<ktt::KernelId> kernelIds;
    };

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

    auto &GetKernelData(std::string className)
    {
      auto it = kernelData.find(className);
      if (kernelData.end() == it) {
        // not found, insert it
        kernelData[className] = {};
        // assume that we don't have too many kernels -> find is cheap
        it = kernelData.find(className);
      }
      return it->second;
    }

    // FIXME add method to access correct stream (and remove interface.GetAllQueues().at(0) from strategy calls)

  private:
    ktt::Tuner tuner;
    std::mutex mutex;
    std::map<std::string, std::map<size_t, KernelIds>> kernelData;
  };

}// namespace utils
}// namespace umpalumpa