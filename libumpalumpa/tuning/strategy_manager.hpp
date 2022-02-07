#pragma once
#include <mutex>
#include <vector>
#include <libumpalumpa/system_includes/ktt.hpp>
// #include <libumpalumpa/tuning/storage.hpp>

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
  std::map<std::string, std::vector<std::shared_ptr<StrategyGroup>>> strategyGroups;
  std::mutex mutex;
  // std::unique_ptr<Storage> tuningData;

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
   * TODO
   */
  void SaveTuningData();

  // template<typename S, typename A>
  // std::vector<std::shared_ptr<StrategyGroup>> LoadTuningData(const S &s, const A &a)
  // {
  //   tuningData->LoadTuningData(s, a);
  // }

  /**
   * TODO
   */
  void Merge(std::vector<std::shared_ptr<StrategyGroup>> &&vec);

  /**
   * Resets the AlgorithmManager, clearing all the saved data (registered strategies, garbage
   * collection metadata).
   */
  void Cleanup();
};

}// namespace umpalumpa::tuning
