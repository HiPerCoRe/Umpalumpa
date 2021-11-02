#include <libumpalumpa/tuning/algorithm_manager.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::algorithm {

AlgorithmManager &AlgorithmManager::Get()
{
  static auto instance = std::unique_ptr<AlgorithmManager>(new AlgorithmManager());
  return *instance;
}

void AlgorithmManager::Register(TunableStrategy &strat)
{
  std::lock_guard<std::mutex> lck(mutex);

  spdlog::info("Strategy at address {0} registered", reinterpret_cast<size_t>(this));// tmp

  for (auto &v : strategies) {
    if (strat.IsSimilarTo(*v[0])) {
      v.push_back(&strat);
      return;
    }
  }
  // 'strat' is not similar to any other registered strategy, add 'strat' to a new vector
  strategies.emplace_back().push_back(&strat);
}

void AlgorithmManager::Unregister(TunableStrategy &strat)
{
  std::lock_guard<std::mutex> lck(mutex);
  // TODO save best configuration

  // FIXME mega in need of refactor
  auto outerIt = strategies.end();
  for (auto &v : strategies) {
    auto it = std::find(v.begin(), v.end(), &strat);
    if (it != v.end()) {
      outerIt = std::find(strategies.begin(), strategies.end(), v);
      v.erase(it);
      spdlog::info("Strategy at address {0} unregistered", reinterpret_cast<size_t>(this));// tmp
      break;
    }
  }
  if (outerIt != strategies.end()) {
    if (outerIt->empty()) { strategies.erase(outerIt); }
  } else {
    spdlog::warn("You are trying to unregister strategy which wasn't previously registered.");// tmp
  }
}

ktt::KernelConfiguration AlgorithmManager::GetBestConfiguration(size_t stratHash)
{
  std::lock_guard<std::mutex> lck(mutex);

  // auto it = strategies.find(stratHash);
  // if (it != strategies.end()) { return it->second->GetBestConfiguration(); }

  // FIXME refactor
  for (auto &v : strategies) {
    for (auto s : v) {
      if (s->GetHash() == stratHash) { return s->GetBestConfiguration(); }
    }
  }

  // TODO Access DB

  return {};// or throw?
}

}// namespace umpalumpa::algorithm

