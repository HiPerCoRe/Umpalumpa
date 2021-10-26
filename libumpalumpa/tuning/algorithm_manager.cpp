#include <libumpalumpa/tuning/algorithm_manager.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <iostream>

namespace umpalumpa::algorithm {

AlgorithmManager &AlgorithmManager::Get()
{
  static auto instance = std::unique_ptr<AlgorithmManager>(new AlgorithmManager());
  return *instance;
}

void AlgorithmManager::Register(TunableStrategy *strat)
{
  std::lock_guard<std::mutex> lck(mutex);
  strategies[strat->GetHash()] = strat;
}
void AlgorithmManager::Unregister(TunableStrategy *strat)
{
  std::lock_guard<std::mutex> lck(mutex);
  // TODO save best configuration

  // Cannot be done using strat->GetHash(), GetHash is not defined during Unregister
  for (auto it = strategies.begin(); it != strategies.end(); ++it) {
    if ((*it).second == strat) {
      strategies.erase(it);
      return;
    }
  }
}

ktt::KernelConfiguration AlgorithmManager::GetBestConfiguration(size_t stratHash)
{
  std::lock_guard<std::mutex> lck(mutex);
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

