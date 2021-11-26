#pragma once
#include <libumpalumpa/system_includes/ktt.hpp>

namespace umpalumpa::utils {
/**
 * Class for tracking KTT ids, that takes care of their proper release when the ids are no longer
 * needed. Tracks all the ids tied to the specified ktt::KernelDefinitionId.
 */
struct KTTIdTracker
{
  /**
   * Creates a KTTIdTracker which tracks all the ids of the specified ktt::KernelDefinitionId.
   */
  KTTIdTracker(ktt::KernelDefinitionId defId, ktt::Tuner &t) : definitionId(defId), tuner(t) {}
  KTTIdTracker(const KTTIdTracker &) = delete;
  KTTIdTracker &operator=(const KTTIdTracker &) = delete;
  KTTIdTracker(KTTIdTracker &&) = delete;
  KTTIdTracker &operator=(KTTIdTracker &&) = delete;

  /**
   * Destroys the KTTIdTracker and releases the tracked ids.
   */
  ~KTTIdTracker()
  {
    // Kernels need to be removed first
    for (auto kId : kernelIds) { tuner.RemoveKernel(kId); }
    // KernelDefinitions need to be removed second
    tuner.RemoveKernelDefinition(definitionId);
    // Arguments need to be removed last
    for (auto aId : argumentIds) { tuner.RemoveArgument(aId); }
  }

  const ktt::KernelDefinitionId definitionId;
  std::vector<ktt::KernelId> kernelIds;
  std::vector<ktt::ArgumentId> argumentIds;// FIXME change to unordered_set

private:
  ktt::Tuner &tuner;
};

}// namespace umpalumpa::utils
