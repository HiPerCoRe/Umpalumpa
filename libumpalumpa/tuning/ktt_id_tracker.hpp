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
    // FIXME turning off logging is just a temporary solution.
    // Under certain conditions KTT logs errors about releasing arguments that are still being used.
    // KTT is right that we are trying to release "still used" arguments, but there is no memory
    // leak, because in the end we release everything properly.
    // This situation occurs when we re-use argument ids for more than 1 kernel definition.
    // Solution for this is not simple because the following situation might happen:
    // Strat1 has defId1, defId2 and argId (argId is used by both definitions). Strat2 has defId1.
    // Now it is impossible to decide which KTTIdTracker should be responsible for releasing argId.
    ktt::Tuner::SetLoggingLevel(ktt::LoggingLevel::Off);
    // Arguments need to be removed last
    for (auto aId : argumentIds) { tuner.RemoveArgument(aId); }
    ktt::Tuner::SetLoggingLevel(ktt::LoggingLevel::Info);
  }

  const ktt::KernelDefinitionId definitionId;
  std::vector<ktt::KernelId> kernelIds;
  std::vector<ktt::ArgumentId> argumentIds;

private:
  ktt::Tuner &tuner;
};

}// namespace umpalumpa::utils
