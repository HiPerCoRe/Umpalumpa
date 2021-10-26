#pragma once
#include <vector>
#include <libumpalumpa/system_includes/ktt.hpp>

namespace umpalumpa::algorithm {

// Forward declarations
class TunableStrategy;

class AlgorithmManager
{
  std::map<size_t, TunableStrategy *> strategies;
  std::vector<ktt::KernelDefinitionId> definitionIds;// Might not be needed

  AlgorithmManager() = default;
  AlgorithmManager(const AlgorithmManager &) = default;
  AlgorithmManager &operator=(const AlgorithmManager &) = default;

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
