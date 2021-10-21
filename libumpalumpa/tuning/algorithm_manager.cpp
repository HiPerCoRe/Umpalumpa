#include <libumpalumpa/tuning/algorithm_manager.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <iostream>

namespace umpalumpa::algorithm {

AlgorithmManager &AlgorithmManager::Get()
{
  static auto instance = std::make_unique<AlgorithmManager>();
  return *instance;
}

void AlgorithmManager::Register(TunableStrategy *strat)
{
  strategies[strat->GetHash()] = strat;
  // std::cout << "Strategy " << strat << " registered\n";
}
void AlgorithmManager::Unregister(TunableStrategy *strat)
{
  // TODO save best configuration

  // Cannot be done using strat->GetHash(), GetHash is not defined during Unregister
  for (auto it = strategies.begin(); it != strategies.end(); ++it) {
    if ((*it).second == strat) {
      strategies.erase(it);
      // std::cout << "Strategy " << strat << " unregistered\n";
      return;
    }
  }
}

ktt::KernelConfiguration AlgorithmManager::GetBestConfiguration(size_t stratHash)
{
  auto it = strategies.find(stratHash);
  if (it != strategies.end()) { return it->second->GetBestConfiguration(); }
  // TODO Access DB
  return {};// or throw?
}
ktt::KernelDefinitionId AlgorithmManager::GetDefinitionId(const std::string & /*sourceFile*/,
  const std::string & /*kernelName*/,
  const std::vector<std::string> & /*templateArgs*/)
{
  return {};
}

}// namespace umpalumpa::algorithm

