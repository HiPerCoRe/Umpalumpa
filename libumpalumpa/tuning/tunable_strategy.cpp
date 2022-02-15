#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::tuning {

TunableStrategy::TunableStrategy(KTTHelper &helper)
  : kttHelper(helper), tuningApproach(TuningApproach::kSelectedKernels),
    canTuneStrategyGroup(false), isRegistered(false), strategyId(GetNewStrategyId())
{}

TunableStrategy::~TunableStrategy()
{
  WaitForKernelsToFinish();
  Cleanup();
}

void TunableStrategy::WaitForKernelsToFinish() const
{
  auto tmp = kttHelper.GetTuner().GetLoggingLevel();
  kttHelper.GetTuner().SetLoggingLevel(ktt::LoggingLevel::Off);
  // We turn logging off because of 'Error' messages that specified actionId wasn't found
  // but that is alright, we don't care about those missing ones we care about those that might
  // still be present
  for (auto actionId : actionIds) { kttHelper.GetTuner().WaitForComputeAction(actionId); }
  kttHelper.GetTuner().SetLoggingLevel(tmp);
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
  if (CanTune(kernelId)) {
    auto tuningResults = RunTuning(kernelId);
    if (tuningResults.IsValid()) {
      SaveTuningToLeader(kernelId, tuningResults);
      // Disable tuning for kernel with id 'kernelId', we already found the best configuration.
      // Do not turn the tuning off if you process the tuning in smaller chunks!
      SetTuningFor(kernelId, false);
    } else {
      spdlog::warn(
        "Tuning result is invalid. Cannot update the best configuration. Will try to run tuning "
        "again in the next strategy execution.");
    }
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
  // Runs of the kernel should be idempotent (user needs to assure this) otherwise the tuning might
  // give the wrong output/results.
  auto results = tuner.Tune(kernelId);
  if (results.empty()) { return ktt::KernelResult(); }// Return invalid result
  return tuner.Run(kernelId, tuner.GetBestConfiguration(kernelId), {});
}

void TunableStrategy::RunBestConfiguration(ktt::KernelId kernelId) const
{
  auto tmp = kttHelper.GetTuner().GetLoggingLevel();
  if (kttLoggingOff) { kttHelper.GetTuner().SetLoggingLevel(ktt::LoggingLevel::Off); }
  kttHelper.GetTuner().Run(kernelId, GetBestConfiguration(kernelId), {});
  if (kttLoggingOff) { kttHelper.GetTuner().SetLoggingLevel(tmp); }
}

void TunableStrategy::SaveTuningToLeader(ktt::KernelId kernelId,
  const ktt::KernelResult &tuningResults)
{
  auto index = GetKernelIndex(kernelId);
  auto bestTimeSoFar = groupLeader->GetBestConfigTime(index);
  // Current implementation of RunTuning method uses a Tuner::Tune method which will run the full
  // tuning and therefore we will only get one KernelResult ever. However, everything is ready to
  // process tuning in smaller chunks (i.e. 100 tuning steps per execution) if the need arises.
  if (tuningResults.GetKernelDuration() < bestTimeSoFar) {
    groupLeader->SetBestConfiguration(index, tuningResults.GetConfiguration());
    groupLeader->SetBestConfigTime(index, tuningResults.GetKernelDuration());
    // If we start processing the tuning in smaller chunks (see ^) the saving of the results should
    // be moved somewhere else because of possible performance drawbacks.
    StrategyManager::Get().SaveTuningData();
  }
}

void TunableStrategy::Register()
{
  StrategyManager::Get().Register(*this);
  isRegistered = true;
}

void TunableStrategy::Cleanup()
{
  if (!idTrackers.empty()) {
    std::lock_guard lck(kttHelper.GetMutex());
    for (auto &sharedTracker : idTrackers) { kttHelper.CleanupIdTracker(sharedTracker); }
  }
  idTrackers.clear();
  definitionIds.clear();
  kernelIds.clear();
  actionIds.clear();
  canTuneStrategyGroup = false;
  if (isRegistered) {
    StrategyManager::Get().Unregister(*this);
    isRegistered = false;
  }
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
