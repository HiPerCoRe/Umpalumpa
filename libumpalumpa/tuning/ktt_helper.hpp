#pragma once

#include <mutex>
#include <libumpalumpa/system_includes/ktt.hpp>
#include <libumpalumpa/utils/cuda.hpp>
#include <libumpalumpa/utils/system.hpp>
#include <libumpalumpa/tuning/ktt_id_tracker.hpp>

namespace umpalumpa::tuning {
class KTTHelper
{
public:
  KTTHelper(const CUcontext context, const std::vector<ktt::ComputeQueue> &queues)
    : tuner(ktt::ComputeApi::CUDA, ktt::ComputeApiInitializer(context, queues))
  {
    tuner.SetCompilerOptions("-I" + utils::kProjectRoot + " --std=c++17 -default-device");
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
    // be careful not to remove queue used by another operation
  }

  // FIXME add method to access correct stream (and remove interface.GetAllQueues().at(0) from
  // strategy calls)

  std::shared_ptr<KTTIdTracker> GetIdTracker(ktt::KernelDefinitionId definitionId)
  {
    if (auto it = tunerIds.find(definitionId); it != tunerIds.end()) { return it->second.lock(); }
    auto sPtr = std::make_shared<KTTIdTracker>(definitionId, tuner);
    tunerIds[definitionId] = sPtr;
    return sPtr;
  }

  void CleanupIdTracker(std::shared_ptr<KTTIdTracker> &trackerSharedPtr)
  {
    auto id = trackerSharedPtr->definitionId;
    trackerSharedPtr.reset();
    if (auto it = tunerIds.find(id); it != tunerIds.end()) {
      if (it->second.expired()) { tunerIds.erase(it); }// erase from map is efficient
    } else {
      // TODO log error, because it HAVE to be present otherwise something is severely wrong
      // we have just a small memory leak, no need to immediately stop execution
    }
  }

private:
  ktt::Tuner tuner;
  std::map<ktt::KernelDefinitionId, std::weak_ptr<KTTIdTracker>> tunerIds;
  std::mutex mutex;
};

}// namespace umpalumpa::tuning
