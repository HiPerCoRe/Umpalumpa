#pragma once
#include <mutex>
#include <vector>
#include <libumpalumpa/system_includes/ktt.hpp>

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

  // FIXME needs to be changed because strategy can have more kernels, therefore there is no single
  // best configuration
  /**
   * Returns the best known configuration of a strategy with the specified hash.
   */
  // ktt::KernelConfiguration GetBestConfiguration(size_t stratHash);

  /**
   * Returns the underlying strategy container.
   *
   * Ideally should be removed in the future.
   */
  const auto &GetRegisteredStrategies() const { return strategies; }

  /**
   * Resets the AlgorithmManager, clearing all the saved data (registered strategies, garbage
   * collection metadata).
   */
  void Cleanup() { strategies.clear(); }
};

}// namespace umpalumpa::algorithm
