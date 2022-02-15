#include <libumpalumpa/tuning/strategy_manager.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>

namespace umpalumpa::tuning {

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

  bool isEqual = false;
  bool isSimilar = false;
  StrategyGroup *groupPtr = nullptr;
  std::string debugMsg = "";

  auto filepath = utils::GetTuningDirectory() + strat.GetUniqueName();
  if (std::filesystem::exists(filepath)) {
    // Works only for equal strategies, similar strategies won't be discovered like this.
    // Unless we update the system for discovering similar strategies we would have to
    // load all the strategies with the name of formatt 'NAME-specificnumbers'
    // where NAME == GetFullName() to find similar ones.
    if (!IsLoaded(filepath)) {
      Merge(strat.LoadTuningData());
      // Mark that we already loaded this file
      loadedFiles[filepath] = true;
    }
    // Use the loaded config, do not tune
    strat.SetTuningApproach(TuningApproach::kNoTuning);
  }

  // Check equality and similarity
  for (auto &group : strategyGroups) {
    // If we find an equal group we are satisfied and we can exit the loop
    // because equality has higher priority than similarity
    if (group.leader->IsEqualTo(strat)) {
      isEqual = true;
      groupPtr = &group;
      break;
    }
    // After we find a similar group we continue looking for an equal group
    // but we ignore any other similar groups
    // TODO might be updated to accept the best of all the similar groups instead of the first one
    if (!isSimilar && group.leader->IsSimilarTo(strat)) {
      isSimilar = true;
      groupPtr = &group;
    }
  }

  if (isEqual) {
    strat.AllowTuningStrategyGroup();
    debugMsg += "As equal to";
  } else if (isSimilar) {
    debugMsg += "As similar to";
  } else {
    // 'strat' does not belong to any of the existing groups, create new group based on the 'strat'
    groupPtr = &strategyGroups.emplace_back(strat);
    // First strategy in a new group can tune the group
    strat.AllowTuningStrategyGroup();
    groupPtr->leader->SetBestConfigurations(strat.GetDefaultConfigurations());
    debugMsg += "As a new Leader";
  }

  strat.AssignLeader(groupPtr->leader.get());
  groupPtr->strategies.push_back(&strat);
  spdlog::debug("Strategy at address {} registered", reinterpret_cast<size_t>(&strat));
  spdlog::debug(debugMsg + " strategy {}", reinterpret_cast<size_t>(groupPtr->leader.get()));
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
      spdlog::debug("Strategy at address {} unregistered", reinterpret_cast<size_t>(&strat));

      // We don't want to remove empty groups... later some strategy may be added there.
      // The group can store best configurations (loaded from db, acquired from KTT, ...), etc...

      return;
    }
  }
  // }

  spdlog::warn("You are trying to unregister strategy which wasn't previously registered.");
}

void StrategyManager::SaveTuningData() const
{
  std::lock_guard lck(mutex);
  // TODO Should be async
  for (const auto &group : strategyGroups) {
    auto filePath = utils::GetTuningDirectory() + group.leader->GetUniqueName();
    std::ofstream outFile(filePath);
    group.SaveTuningData(outFile);
  }
}

void StrategyManager::Merge(StrategyGroup &&newGroup)
{
  for (auto &group : strategyGroups) {
    if (group.IsEqualTo(newGroup)) {
      group.Merge(newGroup);
      return;
    }
  }
  strategyGroups.push_back(std::move(newGroup));
}

void StrategyManager::Cleanup()
{
  std::lock_guard lck(mutex);
  strategyGroups.clear();
}

bool StrategyManager::IsLoaded(const std::string &filename) const
{
  return loadedFiles.find(filename) != loadedFiles.end();
}

}// namespace umpalumpa::tuning

