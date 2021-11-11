#pragma once
#include <mutex>
#include <vector>
#include <libumpalumpa/system_includes/ktt.hpp>
#include <libumpalumpa/tuning/ktt_helper.hpp>
#include <libumpalumpa/tuning/garbage_collector.hpp>

namespace umpalumpa::algorithm {

// Forward declarations
class TunableStrategy;

/**
 * This class groups similar strategies into coherent groups in which the strategies can cooperate
 * on the tuning, or reuse already tuned parameters.
 *
 * Every successfully initialized strategy that utilizes KTT is being automatically registered to
 * the AlgorithmManager. At the end of the strategy's lifetime, it is being automatically
 * unregistered from the AlgorithmManager.
 *
 * AlgorithmManager is a singleton and can be accessed by calling static method
 * AlgorithmManager::Get().
 */
class AlgorithmManager
{
  using StrategyGroup = std::vector<TunableStrategy *>;

  std::vector<StrategyGroup> strategies;
  GarbageCollector garbageCollector;
  std::mutex mutex;

  AlgorithmManager() = default;
  AlgorithmManager(AlgorithmManager &&) = delete;
  AlgorithmManager &operator=(AlgorithmManager &&) = delete;
  // Copy constructor and assign operator are implicitly deleted because of the mutex

public:
  /**
   * Returns an instance of the AlgorithmManager singleton.
   */
  static AlgorithmManager &Get();

  /**
   * Registers the strategy into the AlgorithmManager which allows tuning and usage of prepared
   * tuning parameters.
   */
  void Register(TunableStrategy &strat);

  /**
   * Unregisters the strategy from the AlgorithmManager which disallows tuning and usage of prepared
   * tuning parameters.
   */
  void Unregister(TunableStrategy &strat);

  /**
   * Returns the best known configuration of a strategy with the specified hash.
   */
  ktt::KernelConfiguration GetBestConfiguration(size_t stratHash);

  /**
   * Returns the underlying strategy container.
   *
   * Ideally should be removed in the future.
   */
  auto &GetRegisteredStrategies() const { return strategies; }

  /**
   * Resets the AlgorithmManager, clearing all the saved data (registered strategies, garbage
   * collection metadata).
   */
  void Cleanup();

  /**
   * Returns a KTT's kernel definition Id. If the specified kernel definition already exists,
   * returns the existing Id; otherwise, creates a new one.
   */
  ktt::KernelDefinitionId GetKernelDefinitionId(utils::KTTHelper &kttHelper,
    const std::string &kernelName,
    const std::string &sourceFile,
    const ktt::DimensionVector &gridDimensions,
    const std::vector<std::string> &templateArgs = {});

  /**
   * Registers the argument Ids into garbageCollector.
   */
  void SetKTTArguments(utils::KTTHelper &kttHelper,
    ktt::KernelDefinitionId definitionId,
    const std::vector<ktt::ArgumentId> &argumentIds);

  /**
   * Cleans up provided Ids.
   * If the definition Ids are not used by other strategy removes them from the KTT.
   */
  void CleanupIds(utils::KTTHelper &kttHelper,
    const std::vector<ktt::KernelId> &kernelIds,
    const std::vector<ktt::KernelDefinitionId> &definitionIds);
};

}// namespace umpalumpa::algorithm
