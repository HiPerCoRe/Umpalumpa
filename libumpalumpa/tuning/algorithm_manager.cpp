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
  // too many loops in one function

  for (auto &v : strategies) {
    for (auto *s : v) {
      if (s == &strat) {
        spdlog::warn("You are trying to register the same strategy instance multiple times.");// tmp
        return;
      }
    }
  }

  spdlog::info("Strategy at address {0} registered", reinterpret_cast<size_t>(&strat));// tmp

  // Check equality
  // FIXME probably should iterate through all the strategies?
  for (auto &stratGroup : strategies) {
    if (strat.IsEqualTo(*stratGroup[0])) {
      stratGroup.push_back(&strat);
      spdlog::info("As equal to strategy {0}", reinterpret_cast<size_t>(stratGroup[0]));// tmp
      // TODO set additional flags (i.e. this strategy can be used for tuning)
      return;
    }
  }

  // Check similarity
  for (auto &stratGroup : strategies) {
    if (strat.IsSimilarTo(*stratGroup[0])) {
      stratGroup.push_back(&strat);
      spdlog::info("As similar to strategy {0}", reinterpret_cast<size_t>(stratGroup[0]));// tmp
      return;
    }
  }

  // 'strat' is not equal or similar to any other registered strategy, add 'strat' to a new vector
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
      spdlog::info("Strategy at address {0} unregistered", reinterpret_cast<size_t>(&strat));// tmp

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

void AlgorithmManager::Cleanup()
{
  // Causes deadlock because during destruction we call Unregister
  // std::lock_guard<std::mutex> lck(mutex);

  // Properly calls garbageCollector and cleans up the KTT ids via ~TunableStrategy
  strategies.clear();
}

ktt::KernelDefinitionId AlgorithmManager::GetKernelDefinitionId(utils::KTTHelper &kttHelper,
  const std::string &kernelName,
  const std::string &sourceFile,
  const ktt::DimensionVector &gridDimensions,
  const std::vector<std::string> &templateArgs)
{
  std::lock_guard<std::mutex> lck(mutex);

  auto &tuner = kttHelper.GetTuner();
  auto id = tuner.GetKernelDefinitionId(kernelName, templateArgs);
  if (id == ktt::InvalidKernelDefinitionId) {
    id =
      tuner.AddKernelDefinitionFromFile(kernelName, sourceFile, gridDimensions, {}, templateArgs);
  }

  garbageCollector.RegisterKernelDefinitionId(id, kttHelper);
  return id;
}

void AlgorithmManager::CleanupIds(utils::KTTHelper &kttHelper,
  const std::vector<ktt::KernelId> &kernelIds,
  const std::vector<ktt::KernelDefinitionId> &definitionIds)
{
  garbageCollector.CleanupIds(kttHelper, kernelIds, definitionIds);
}

void AlgorithmManager::SetKTTArguments(utils::KTTHelper &kttHelper,
  ktt::KernelDefinitionId definitionId,
  const std::vector<ktt::ArgumentId> &argumentIds)
{
  garbageCollector.RegisterArgumentIds(definitionId, argumentIds, kttHelper);
}

}// namespace umpalumpa::algorithm

