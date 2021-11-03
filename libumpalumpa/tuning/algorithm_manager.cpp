#include <algorithm>
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

  // FIXME refactor
  for (auto &v : strategies) {
    for (auto *s : v) {
      if (s == &strat) {
        spdlog::warn("You are trying to register the same strategy multiple times.");// tmp
        return;
      }
    }
  }

  spdlog::info("Strategy at address {0} registered", reinterpret_cast<size_t>(this));// tmp

  for (auto &stratGroup : strategies) {
    if (strat.IsSimilarTo(*stratGroup[0])) {
      stratGroup.push_back(&strat);
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

  for (auto &stratGroup : strategies) {
    auto stratIt = std::find(stratGroup.begin(), stratGroup.end(), &strat);

    if (stratIt != stratGroup.end()) {
      // Remove strategy from group
      std::iter_swap(stratIt, stratGroup.end() - 1);
      stratGroup.pop_back();
      spdlog::info("Strategy at address {0} unregistered", reinterpret_cast<size_t>(this));// tmp

      // Remove empty group
      if (stratGroup.empty()) {
        std::iter_swap(&stratGroup, strategies.end() - 1);
        strategies.pop_back();
      }
      return;
    }
  }

  spdlog::warn("You are trying to unregister strategy which wasn't previously registered.");// tmp
}

ktt::KernelConfiguration AlgorithmManager::GetBestConfiguration(size_t stratHash)
{
  std::lock_guard<std::mutex> lck(mutex);

  // FIXME refactor
  for (auto &stratGroup : strategies) {
    for (auto s : stratGroup) {
      if (s->GetHash() == stratHash) { return s->GetBestConfiguration(); }
    }
  }

  // TODO Access DB

  return {};// or throw?
}

}// namespace umpalumpa::algorithm

