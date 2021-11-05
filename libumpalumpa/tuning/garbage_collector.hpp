#pragma once
#include <map>
#include <libumpalumpa/tuning/ktt_helper.hpp>

namespace umpalumpa {

class GarbageCollector
{
  using KTTIdentifier = std::reference_wrapper<utils::KTTHelper>;

  class IdTracker
  {
    struct DefinitionData
    {
      size_t referenceCounter = 0;
      std::vector<ktt::ArgumentId> arguments;
    };

  public:
    std::map<ktt::KernelDefinitionId, DefinitionData> data;
    utils::KTTHelper &kttHelper;

    IdTracker(utils::KTTHelper &helper) : kttHelper(helper) {}

    void CleanupDefinitionId(ktt::KernelDefinitionId id)
    {
      kttHelper.GetTuner().RemoveKernelDefinition(id);
    }

    void CleanupArguments(ktt::KernelDefinitionId id)
    {
      for (auto arg : data.at(id).arguments) { kttHelper.GetTuner().RemoveArgument(arg); }
    }

    void Cleanup(ktt::KernelDefinitionId id)
    {
      data.at(id).referenceCounter--;
      if (data.at(id).referenceCounter == 0u) {
        // Order of cleanup is important
        CleanupDefinitionId(id);
        CleanupArguments(id);
        data.erase(id);
      }
    }
  };

  // We need to distinguish ids of different KTTs
  std::map<KTTIdentifier, IdTracker> kttIds;

public:
  void CleanupIds(KTTIdentifier kttIdentifier,
    const std::vector<ktt::KernelId> &kernelIds,
    const std::vector<ktt::KernelDefinitionId> &definitionIds)
  {
    auto &tracker = kttIds.at(kttIdentifier);
    std::lock_guard<std::mutex> lck(tracker.kttHelper.GetMutex());

    // Order of cleanup is important
    for (auto kId : kernelIds) { tracker.kttHelper.GetTuner().RemoveKernel(kId); }
    for (auto dId : definitionIds) { tracker.Cleanup(dId); }
  }

  void RegisterKernelDefinitionId(ktt::KernelDefinitionId id, KTTIdentifier kttIdentifier)
  {
    kttIds.try_emplace(kttIdentifier, kttIdentifier.get());
    // If 'id' isn't present yet, creates a new DefinitionData
    // In the end increases the reference counter
    kttIds.at(kttIdentifier).data[id].referenceCounter++;
  }

  void RegisterArgumentIds(ktt::KernelDefinitionId definitionId,
    const std::vector<ktt::ArgumentId> &argumentIds,
    KTTIdentifier kttIdentifier)
  {
    auto &v = kttIds.at(kttIdentifier).data.at(definitionId).arguments;
    v.insert(v.end(), argumentIds.begin(), argumentIds.end());
  }
};

}// namespace umpalumpa
