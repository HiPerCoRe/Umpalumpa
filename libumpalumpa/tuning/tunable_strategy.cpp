#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::tuning {

TunableStrategy::TunableStrategy(KTTHelper &helper)
  : kttHelper(helper), tuningApproach(TuningApproach::kNoTuning), canTuneStrategyGroup(false),
    isRegistered(false), strategyId(GetNewStrategyId())
{}

TunableStrategy::~TunableStrategy()
{
  // FIXME Needs to be synchronized
  // kttHelper.GetTuner().Synchronize();
  if (isRegistered) { StrategyManager::Get().Unregister(*this); }
  if (!idTrackers.empty()) {
    // Needs to be locked because Cleanup routine accesses ktt::Tuner
    std::lock_guard lck(kttHelper.GetMutex());
    Cleanup();
  }
}

// This version is currently unused, might be removed later
bool TunableStrategy::IsEqualTo(const TunableStrategy &ref) const
{
  return GetHash() == ref.GetHash();
}

std::string TunableStrategy::GetFullName() const { return typeid(*this).name(); }

ktt::KernelConfiguration TunableStrategy::GetBestConfiguration(ktt::KernelId kernelId) const
{
  return GetBestConfigurations().at(GetKernelIndex(kernelId));
}

const std::vector<ktt::KernelConfiguration> &TunableStrategy::GetBestConfigurations() const
{
  if (groupLeader != nullptr) { return groupLeader->GetBestConfigurations(); }
  throw std::logic_error(
    "You are trying to access StrategyGroup Leader from a strategy that does not belong to any "
    "StrategyGroup!");
}

bool TunableStrategy::ShouldBeTuned(ktt::KernelId kernelId) const
{
  switch (tuningApproach) {
  case TuningApproach::kEntireStrategy:
    return true;
  case TuningApproach::kSelectedKernels:
    return kernelIds.at(GetKernelIndex(kernelId)).tune;
  case TuningApproach::kNoTuning:
    [[fallthrough]];
  default:
    return false;
  }
}

void TunableStrategy::ExecuteKernel(ktt::KernelId kernelId)
{
  if (canTuneStrategyGroup && ShouldBeTuned(kernelId)) {
    auto tuningResults = RunTuning(kernelId);
    SaveTuningToLeader(kernelId, tuningResults);
  } else {
    RunBestConfiguration(kernelId);
  }
}

ktt::KernelResult TunableStrategy::RunTuning(ktt::KernelId kernelId) const
{
  auto &tuner = kttHelper.GetTuner();
  // TODO
  // We need to let the rest of the kernels finish, while we won't allow anyone to start a new
  // kernel (this is done by locking the Tuner in Execute method).
  // tuner.Synchronize();
  // Now, there are no kernels at the GPU and we can start tuning
  auto results = tuner.TuneIteration(kernelId, {});
  tuner.SynchronizeDevice();// tmp solution to make the call blocking
  // TODO run should be blocking while tuning -> need change in the KernelLauncher
  return results;
}

void TunableStrategy::RunBestConfiguration(ktt::KernelId kernelId) const
{
  kttHelper.GetTuner().Run(kernelId, GetBestConfiguration(kernelId), {});
}

void TunableStrategy::SaveTuningToLeader(ktt::KernelId kernelId,
  const ktt::KernelResult &tuningResults)
{
  auto index = GetKernelIndex(kernelId);
  auto bestTimeSoFar = groupLeader->GetBestConfigTime(index);
  if (tuningResults.GetKernelDuration() < bestTimeSoFar) {
    groupLeader->SetBestConfiguration(index, kttHelper.GetTuner().GetBestConfiguration(kernelId));
    groupLeader->SetBestConfigTime(index, tuningResults.GetKernelDuration());
  }
}

void TunableStrategy::Register()
{
  StrategyManager::Get().Register(*this);
  isRegistered = true;
}

void TunableStrategy::Cleanup()
{
  for (auto &sharedTracker : idTrackers) { kttHelper.CleanupIdTracker(sharedTracker); }
  idTrackers.clear();
  definitionIds.clear();
  kernelIds.clear();
  canTuneStrategyGroup = false;
  // FIXME needs to unregister aswell!!!
  isRegistered = false;
}

void TunableStrategy::AddKernelDefinition(const std::string &kernelName,
  const std::string &sourceFile,
  const ktt::DimensionVector &gridDimensions,
  const std::vector<std::string> &templateArgs)
{
  auto &tuner = kttHelper.GetTuner();
  auto id = tuner.GetKernelDefinitionId(kernelName, templateArgs);
  if (id == ktt::InvalidKernelDefinitionId) {
    id =
      tuner.AddKernelDefinitionFromFile(kernelName, sourceFile, gridDimensions, {}, templateArgs);
  }

  if (id == ktt::InvalidKernelDefinitionId) {
    throw std::invalid_argument("Definition id could not be created.");
  }

  definitionIds.push_back(id);
  idTrackers.push_back(kttHelper.GetIdTracker(id));
}

void TunableStrategy::AddKernel(const std::string &name, ktt::KernelDefinitionId definitionId)
{
  auto kernelId =
    kttHelper.GetTuner().CreateSimpleKernel(name + "_" + std::to_string(strategyId), definitionId);

  if (kernelId == ktt::InvalidKernelId) {
    throw std::invalid_argument("Kernel id could not be created.");
  }

  kernelIds.push_back({ kernelId });
  idTrackers.at(GetDefinitionIndex(definitionId))->kernelIds.push_back(kernelId);
}

void TunableStrategy::SetArguments(ktt::KernelDefinitionId id,
  const std::vector<ktt::ArgumentId> &argumentIds)
{
  kttHelper.GetTuner().SetArguments(id, argumentIds);
  auto &tmp = idTrackers.at(GetDefinitionIndex(id))->argumentIds;
  tmp.insert(tmp.end(), argumentIds.begin(), argumentIds.end());
}

size_t TunableStrategy::GetDefinitionIndex(ktt::KernelDefinitionId id) const
{
  return static_cast<size_t>(std::distance(
    definitionIds.begin(), std::find(definitionIds.begin(), definitionIds.end(), id)));
}

size_t TunableStrategy::GetKernelIndex(ktt::KernelId id) const
{
  return static_cast<size_t>(std::distance(kernelIds.begin(),
    std::find_if(kernelIds.begin(), kernelIds.end(), [id](auto &x) { return x.id == id; })));
}

size_t TunableStrategy::GetNewStrategyId()
{
  static std::mutex mutex;
  static size_t strategyCounter = 1;
  std::lock_guard<std::mutex> lck(mutex);
  return strategyCounter++;
}

}// namespace umpalumpa::tuning
