#include <libumpalumpa/tuning/algorithm_manager.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>

namespace umpalumpa::algorithm {

AlgorithmManager &AlgorithmManager::Get()
{
  static auto instance = std::unique_ptr<AlgorithmManager>(new AlgorithmManager());
  return *instance;
}

void AlgorithmManager::Register(TunableStrategy &strat)
{
  std::lock_guard lck(mutex);

  // FIXME refactor
  // too many loops in one function

  for (auto &group : strategyGroups) {
    for (auto *s : group.strategies) {
      if (s == &strat) {
        spdlog::warn("You are trying to register the same strategy instance multiple times.");
        return;
      }
    }
  }

  spdlog::debug("Strategy at address {0} registered", reinterpret_cast<size_t>(&strat));

  // Check equality
  for (auto &group : strategyGroups) {
    if (group.leader->IsEqualTo(strat)) {
      group.strategies.push_back(&strat);
      spdlog::debug("As equal to strategy {0}", reinterpret_cast<size_t>(group.leader.get()));
      strat.AllowTuningStrategyGroup();
      strat.groupLeader = group.leader.get();
      return;
    }
  }

  // Check similarity
  for (auto &group : strategyGroups) {
    if (group.leader->IsSimilarTo(strat)) {
      group.strategies.push_back(&strat);
      spdlog::debug("As similar to strategy {0}", reinterpret_cast<size_t>(group.leader.get()));
      strat.groupLeader = group.leader.get();
      return;
    }
  }

  // 'strat' is not equal or similar to any other registered strategy, add 'strat' to a new group
  strategyGroups.emplace_back(strat).strategies.push_back(&strat);
  // First strategy in a new group can tune the group
  strat.AllowTuningStrategyGroup();
  auto &group = strategyGroups.back();
  strat.groupLeader = group.leader.get();
  group.leader->SetBestConfigurations(strat.GetDefaultConfigurations());
}

void AlgorithmManager::Unregister(TunableStrategy &strat)
{
  std::lock_guard lck(mutex);
  // TODO save best configuration

  for (auto &group : strategyGroups) {
    auto stratIt = std::find(group.strategies.begin(), group.strategies.end(), &strat);

    if (stratIt != group.strategies.end()) {
      // Remove strategy from group
      std::iter_swap(stratIt, group.strategies.end() - 1);
      group.strategies.pop_back();
      spdlog::debug("Strategy at address {0} unregistered", reinterpret_cast<size_t>(&strat));

      // We don't want to remove empty groups... later some strategy may be added there.
      // The group can store best configurations (loaded from db, acquired from KTT, ...), etc...

      return;
    }
  }

  spdlog::warn("You are trying to unregister strategy which wasn't previously registered.");
}

void AlgorithmManager::Cleanup()
{
  std::lock_guard lck(mutex);
  strategyGroups.clear();
}

const std::vector<ktt::KernelConfiguration> &AlgorithmManager::GetBestConfigurations(
  size_t stratHash)
{
  std::lock_guard lck(mutex);

  // Assuming that only Strategy that is currently registered in the AlgorithmManager can ask for
  // the best configuration
  for (auto &group : strategyGroups) {
    for (auto s : group.strategies) {
      if (s->GetHash() == stratHash) { return group.leader->GetBestConfigurations(); }
    }
  }

  spdlog::warn("There is no configuration for a strategy with the hash {0}", stratHash);
  throw std::invalid_argument("");
  // return {};
}

}// namespace umpalumpa::algorithm

