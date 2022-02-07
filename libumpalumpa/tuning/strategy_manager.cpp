#include <libumpalumpa/tuning/strategy_manager.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>
// #include <libumpalumpa/tuning/storage.hpp>

namespace umpalumpa::tuning {

StrategyManager &StrategyManager::Get()
{
  static auto instance = std::unique_ptr<StrategyManager>(new StrategyManager());
  return *instance;
}

void StrategyManager::Register(TunableStrategy &strat)
{
  std::lock_guard lck(mutex);

  auto &specificGroups = strategyGroups[strat.GetFullName()];

  for (auto &group : specificGroups) {
    for (auto *s : group->strategies) {
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

  // if (!tuningData->IsLoaded(strat.GetFullName())) { Merge(strat.LoadTuningData()); }
  Merge(strat.LoadTuningData());

  // Check equality and similarity
  for (auto &group : specificGroups) {
    // If we find an equal group we are satisfied and we can exit the loop
    // because equality has higher priority than similarity
    if (group->leader->IsEqualTo(strat)) {
      isEqual = true;
      groupPtr = group.get();
      break;
    }
    // After we find a similar group we continue looking for an equal group
    // but we ignore any other similar groups
    // TODO might be updated to accept the best of all the similar groups instead of the first one
    if (!isSimilar && group->leader->IsSimilarTo(strat)) {
      isSimilar = true;
      groupPtr = group.get();
    }
  }

  if (isEqual) {
    strat.AllowTuningStrategyGroup();
    debugMsg += "As equal to";
  } else if (isSimilar) {
    debugMsg += "As similar to";
  } else {
    // 'strat' does not belong to any of the existing groups, create new group based on the 'strat'
    groupPtr = specificGroups.emplace_back(std::make_shared<StrategyGroup>(strat)).get();
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

  // FIXME doesn't work, because this method is called from the destructor of TunableStrategy
  // auto &specificGroups = strategyGroups[strat.GetFullName()];

  for (auto &[_, specificGroups] : strategyGroups) {// viz fixme ^, should be removed
    for (auto &group : specificGroups) {
      auto stratIt = std::find(group->strategies.begin(), group->strategies.end(), &strat);

      if (stratIt != group->strategies.end()) {
        // Remove strategy from group
        std::iter_swap(stratIt, group->strategies.end() - 1);
        group->strategies.pop_back();
        spdlog::debug("Strategy at address {} unregistered", reinterpret_cast<size_t>(&strat));

        // We don't want to remove empty groups... later some strategy may be added there.
        // The group can store best configurations (loaded from db, acquired from KTT, ...), etc...

        return;
      }
    }
  }

  spdlog::warn("You are trying to unregister strategy which wasn't previously registered.");
}

void StrategyManager::SaveTuningData()
{
  // TODO Should be async
  for (const auto &[name, groups] : strategyGroups) {
    if (groups.empty()) { continue; }
    std::ofstream outFile(utils::GetTuningDirectory() + name);
    for (const auto &group : groups) { group->Serialize(outFile); }
  }
  // tuningData->Save();
}

void StrategyManager::Merge(std::vector<std::shared_ptr<StrategyGroup>> &&loadedSG)
{
  for (auto &newGroup : loadedSG) {
    bool addNewGroup = true;
    auto &specificGroups = strategyGroups[newGroup->leader->GetFullName()];

    for (auto &group : specificGroups) {
      if (group->IsEqualTo(*newGroup)) {
        group->Merge(*newGroup);
        addNewGroup = false;
        break;
      }
    }
    if (addNewGroup) { specificGroups.push_back(std::move(newGroup)); }
  }
}

void StrategyManager::Cleanup()
{
  std::lock_guard lck(mutex);
  strategyGroups.clear();
}

}// namespace umpalumpa::tuning

