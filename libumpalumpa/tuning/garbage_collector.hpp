#pragma once
#include <map>
#include <mutex>
#include <atomic>
#include <libumpalumpa/tuning/ktt_helper.hpp>

namespace umpalumpa {

/**
 * This class takes care of cleaning a memory of all the registered KTT ids.
 */
class GarbageCollector
{
  // KTTHelper doesn't have anything else that can be used as a key in a map.
  using KTTIdentifier = utils::KTTHelper *;

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
      std::atomic<size_t> referenceCounter = 0;
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

    /**
     * Returns true when the IdTracker doesn't contain any data.
     */
    bool IsEmpty() const { return data.empty(); }
  };

  // We need to distinguish ids of different KTTs
  std::map<KTTIdentifier, IdTracker> kttIds;

  // FIXME don't lock entire GarbageCollector, different KTTs can be cleaned concurrently
  std::mutex mutex;

  /**
   * Registers KTTHelper and its KTT tuner for tracking the ids, if it wasn't registered before.
   * Returns KTTIdentifier used for accessing the correct IdTracker.
   */
  KTTIdentifier GetIdentifier(utils::KTTHelper &kttHelper)
  {
    kttIds.try_emplace(&kttHelper, kttHelper);
    return &kttHelper;
  }

public:
  /**
   * Removes the specified kernel ids from KTT and decreases reference counter for the specified
   * definition ids. If for some definition id the reference counter drops to zero, the definition
   * id is removed from KTT.
   */
  void CleanupIds(utils::KTTHelper &kttHelper,
    const std::vector<ktt::KernelId> &kernelIds,
    const std::vector<ktt::KernelDefinitionId> &definitionIds)
  {
    std::lock_guard<std::mutex> lck(mutex);// FIXME see comment at mutex declaration
    auto kttIdentifier = GetIdentifier(kttHelper);
    auto &tracker = kttIds.at(kttIdentifier);
    // We lock access to the KTT while we clean up (to not interleave insertion and removal)
    std::lock_guard<std::mutex> kttLck(tracker.kttHelper.GetMutex());

    // Order of cleanup is important
    for (auto kId : kernelIds) { tracker.kttHelper.GetTuner().RemoveKernel(kId); }
    for (auto dId : definitionIds) { tracker.Cleanup(dId); }
    // if (tracker.IsEmpty()) { kttIds.erase(kttIdentifier); }// maybe not needed
  }

  /**
   * Increases the reference counter for the specified kernel definition id.
   * If the specified id is not being tracked yet, it gets created.
   */
  void RegisterKernelDefinitionId(ktt::KernelDefinitionId id, utils::KTTHelper &kttHelper)
  {
    std::lock_guard<std::mutex> lck(mutex);// FIXME see comment at mutex declaration
    auto kttIdentifier = GetIdentifier(kttHelper);
    // If 'id' isn't present yet, creates a new DefinitionData
    // In the end increases the reference counter
    kttIds.at(kttIdentifier).data[id].referenceCounter++;
  }

  /**
   * Registers the argument ids to the specified definition id.
   */
  void RegisterArgumentIds(ktt::KernelDefinitionId definitionId,
    const std::vector<ktt::ArgumentId> &argumentIds,
    utils::KTTHelper &kttHelper)
  {
    std::lock_guard<std::mutex> lck(mutex);// FIXME see comment at mutex declaration
    auto kttIdentifier = GetIdentifier(kttHelper);
    auto &v = kttIds.at(kttIdentifier).data.at(definitionId).arguments;
    v.insert(v.end(), argumentIds.begin(), argumentIds.end());
  }
};

}// namespace umpalumpa
