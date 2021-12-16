#include <libumpalumpa/tuning/strategy_manager.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>

namespace umpalumpa::algorithm {

StrategyManager &StrategyManager::Get()
{
  static auto instance = std::unique_ptr<StrategyManager>(new StrategyManager());
  return *instance;
}

void StrategyManager::Register(TunableStrategy &strat)
{
  std::lock_guard lck(mutex);

  for (auto &group : strategyGroups) {
    for (auto *s : group.strategies) {
      if (s == &strat) {
        spdlog::warn("You are trying to register the same strategy instance multiple times.");
        return;
      }
    }
  }

  bool equal = false;
  bool similar = false;
  StrategyGroup *groupPtr = nullptr;
  std::string debugMsg = "";

  // Check equality and similarity
  for (auto &group : strategyGroups) {
    if (group.leader->IsEqualTo(strat)) {
      equal = true;
      groupPtr = &group;
      break;
    }
    if (!similar && group.leader->IsSimilarTo(strat)) {
      similar = true;
      groupPtr = &group;
    }
  }

  if (equal) {
    strat.AllowTuningStrategyGroup();
    debugMsg += "As equal to";
  } else if (similar) {
    debugMsg += "As similar to";
  } else {
    // 'strat' does not belong to any of the existing groups, create new group based on the 'strat'
    groupPtr = &strategyGroups.emplace_back(strat);
    // First strategy in a new group can tune the group
    strat.AllowTuningStrategyGroup();
    groupPtr->leader->SetBestConfigurations(strat.GetDefaultConfigurations());
    debugMsg += "As new Leader";
  }

  strat.groupLeader = groupPtr->leader.get();
  groupPtr->strategies.push_back(&strat);
  spdlog::debug("Strategy at address {0} registered", reinterpret_cast<size_t>(&strat));
  spdlog::debug(debugMsg + " strategy {0}", reinterpret_cast<size_t>(groupPtr->leader.get()));
}

void StrategyManager::Unregister(TunableStrategy &strat)
{
  std::lock_guard lck(mutex);

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

void StrategyManager::Cleanup()
{
  std::lock_guard lck(mutex);
  strategyGroups.clear();
}

}// namespace umpalumpa::algorithm

