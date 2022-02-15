#pragma once
#include <mutex>
#include <vector>
#include <libumpalumpa/system_includes/ktt.hpp>

namespace umpalumpa::tuning {

// Forward declarations
class TunableStrategy;
struct StrategyGroup;

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
class StrategyManager
{
  std::vector<StrategyGroup> strategyGroups;
  mutable std::mutex mutex;
  std::map<std::string, bool> loadedFiles;

  StrategyManager() = default;
  StrategyManager(StrategyManager &&) = delete;
  StrategyManager &operator=(StrategyManager &&) = delete;
  // Copy constructor and assign operator are implicitly deleted because of the mutex

public:
  /**
   * Returns an instance of the AlgorithmManager singleton.
   */
  static StrategyManager &Get();

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
   * Returns the underlying strategy container.
   *
   * Ideally should be removed in the future.
   */
  const auto &GetRegisteredStrategies() const { return strategyGroups; }

  /**
   * Saves tuning data of all the strategy groups.
   * TODO should be async
   */
  void SaveTuningData() const;

  /**
   * Resets the AlgorithmManager, clearing all the saved data (registered strategies, garbage
   * collection metadata).
   */
  void Cleanup();

  /**
   * Returns true when the specified file has been loaded during this runtime.
   */
  bool IsLoaded(const std::string &filename) const;

protected:
  /**
   * Merges provided strategy group into the strategy groups saved in the StrategyManager.
   * Makes sure to remove duplicity. In case of found duplicity, keeps the strategy group instance
   * with better (faster) kernel execution time.
   */
  void Merge(StrategyGroup &&vec);
};

}// namespace umpalumpa::tuning
