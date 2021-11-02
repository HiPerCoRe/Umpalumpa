#pragma once
#include <mutex>
#include <vector>
#include <libumpalumpa/system_includes/ktt.hpp>

namespace umpalumpa::algorithm {

// Forward declarations
class TunableStrategy;

class AlgorithmManager
{
  std::vector<std::vector<TunableStrategy *>> strategies;
  std::mutex mutex;

  AlgorithmManager() = default;
  AlgorithmManager(AlgorithmManager &&) = delete;
  AlgorithmManager &operator=(AlgorithmManager &&) = delete;
  // Copy constructor and assign operator are implicitly deleted because of the mutex

public:
  static AlgorithmManager &Get();

  void Register(TunableStrategy &strat);
  void Unregister(TunableStrategy &strat);
  ktt::KernelConfiguration GetBestConfiguration(size_t stratHash);
  const auto &GetRegisteredStrategies() const { return strategies; }
  void Reset() { strategies.clear(); }
};

}// namespace umpalumpa::algorithm
