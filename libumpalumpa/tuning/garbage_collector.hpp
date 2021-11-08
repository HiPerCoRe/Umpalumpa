#pragma once
#include <map>
#include <libumpalumpa/tuning/ktt_helper.hpp>

namespace umpalumpa {

/**
 * This class takes care of cleaning a memory of all the registered KTT ids.
 */
class GarbageCollector
{
  using KTTIdentifier = std::reference_wrapper<utils::KTTHelper>;

  /**
   * Internal class used for tracking and removing the KTT ids.
   */
  class IdTracker
  {
    /**
     * Internal structure for storing metadata about the definition ids.
     */
    struct DefinitionData
    {
      size_t referenceCounter = 0;
      std::vector<ktt::ArgumentId> arguments;
    };

  public:
    std::map<ktt::KernelDefinitionId, DefinitionData> data;
    utils::KTTHelper &kttHelper;

    /**
     * Creates the IdTracker for the KTT tuner of the specified KTTHelper.
     */
    IdTracker(utils::KTTHelper &helper) : kttHelper(helper) {}

    /**
     * Removes the specified kernel definition id from the KTT.
     * Any kernel id that was previously tied to the specified kernel definition id must be removed
     * from the KTT before calling this method.
     */
    void CleanupDefinitionId(ktt::KernelDefinitionId id)
    {
      kttHelper.GetTuner().RemoveKernelDefinition(id);
    }

    /**
     * Removes all argument ids tied to the specified kernel definition id.
     * The specified kernel definition id needs to be already removed from the KTT before calling
     * this method.
     */
    void CleanupArguments(ktt::KernelDefinitionId id)
    {
      for (auto arg : data.at(id).arguments) { kttHelper.GetTuner().RemoveArgument(arg); }
    }

    /**
     * Performs clean up of the specified kernel definition id.
     */
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
  /**
   * Removes the specified kernel ids from KTT and decreases reference counter for the specified
   * definition ids. If for some definition id the reference counter drops to zero, the definition
   * id is removed from KTT.
   */
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

  /**
   * Increases the reference counter for the specified kernel definition id.
   * If the specified id is not being tracked yet, it gets created.
   */
  void RegisterKernelDefinitionId(ktt::KernelDefinitionId id, KTTIdentifier kttIdentifier)
  {
    kttIds.try_emplace(kttIdentifier, kttIdentifier.get());
    // If 'id' isn't present yet, creates a new DefinitionData
    // In the end increases the reference counter
    kttIds.at(kttIdentifier).data[id].referenceCounter++;
  }

  /**
   * Registers the argument ids to the specified definition id.
   */
  void RegisterArgumentIds(ktt::KernelDefinitionId definitionId,
    const std::vector<ktt::ArgumentId> &argumentIds,
    KTTIdentifier kttIdentifier)
  {
    auto &v = kttIds.at(kttIdentifier).data.at(definitionId).arguments;
    v.insert(v.end(), argumentIds.begin(), argumentIds.end());
  }
};

}// namespace umpalumpa
