#pragma once
#include <mutex>
#include <vector>
#include <libumpalumpa/system_includes/ktt.hpp>

namespace umpalumpa::algorithm {

// Forward declarations
class TunableStrategy;

class AlgorithmManager
{
  std::map<size_t, TunableStrategy *> strategies;
  std::vector<ktt::KernelDefinitionId> definitionIds;// Might not be needed
  std::mutex mutex;

  AlgorithmManager() = default;
  AlgorithmManager(AlgorithmManager &&) = delete;
  AlgorithmManager &operator=(AlgorithmManager &&) = delete;
  // Copy constructor and assign operator are implicitly deleted because of the mutex


public:
  static AlgorithmManager &Get();

  void Register(TunableStrategy *strat);
  void Unregister(TunableStrategy *strat);
  ktt::KernelConfiguration GetBestConfiguration(size_t stratHash);
  ktt::KernelDefinitionId GetDefinitionId(const std::string &sourceFile,
    const std::string &kernelName,
    const std::vector<std::string> &templateArgs);
};

}// namespace umpalumpa::algorithm
