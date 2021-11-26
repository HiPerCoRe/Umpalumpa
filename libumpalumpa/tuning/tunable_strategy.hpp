#pragma once
#include <libumpalumpa/tuning/ktt_helper.hpp>
#include <libumpalumpa/tuning/algorithm_manager.hpp>

namespace umpalumpa::algorithm {

/**
 * Base class for every strategy that uses KTT for tuning.
 * Having this class as a predecessor automates many tasks tied to the tuning process.
 */
class TunableStrategy
{
public:
  /**
   * Creates and initializes TunableStrategy.
   */
  TunableStrategy(utils::KTTHelper &helper)
    : kttHelper(helper), strategyId(GetNewStrategyId()), tune(false), isRegistered(false)
  {}

  /**
   * Destroys the TunableStrategy. Cleans up all the resources (KTT ids) tied to this instance.
   */
  virtual ~TunableStrategy()
  {
    // FIXME Needs to be synchronized
    // kttHelper.GetTuner().Synchronize();
    if (isRegistered) { AlgorithmManager::Get().Unregister(*this); }
    if (!idTrackers.empty()) {
      // Needs to be locked because Cleanup routine accesses ktt::Tuner
      std::lock_guard lck(kttHelper.GetMutex());
      Cleanup();
    }
  }

  /**
   * Creates a Leader strategy out of this strategy.
   * In order to get the correct type of the strategy, this method needs to be overriden by the
   * successor classes it the following way:
   *
   * return algorithm::StrategyGroup::CreateLeader(*this, alg);
   */
  virtual std::unique_ptr<TunableStrategy> CreateLeader() const = 0;

  /**
   * Returns hash of this strategy. This method needs to be overriden by successor strategy because
   * the hash is computed using algorithm specific data and settings.
   */
  virtual size_t GetHash() const = 0;

  /**
   * Each successor strategy defines similarity by its own rules.
   * When we have two similar strategies, we can reuse tuning parameters of one strategy when
   * executing the other one.
   */
  virtual bool IsSimilarTo(const TunableStrategy &ref) const = 0;

  /**
   * Two strategies are equal when their hashes are the same.
   * When we have two equal strategies, we can use both for tuning.
   */
  bool IsEqualTo(const TunableStrategy &ref) const { return GetHash() == ref.GetHash(); }

  /**
   * Returns the full name of the strategy type (including namespaces).
   */
  std::string GetFullName() const { return typeid(*this).name(); }

  /**
   * Returns the best known tuning configuration of the specified kernel saved in the KTT tuner.
   */
  ktt::KernelConfiguration GetBestConfiguration(ktt::KernelId kernelId) const
  {
    return kttHelper.GetTuner().GetBestConfiguration(kernelId);
  }

  /**
   * Sets a flag that controls whether this strategy should be tuned or not.
   */
  void SetTuning(bool val) { tune = val; }

  /**
   * Returns a flag that that controls whether this strategy should be tuned or not.
   */
  bool ShouldTune() { return tune; }

  /**
   * Runs a tuning of the specified kernel.
   * TODO In order to correctly evaluate the tuning, this method waits until all the currently
   * running kernels finish.
   *
   * TODO Execution of this method is blocking because we need to stay in the critical section for
   * the entire duration of tuning.
   *
   * TODO IMPORTANT: This method assumes that it is being called from a critical section which locks
   * the KTT tuner.
   */
  void RunTuning(ktt::KernelId kernelId) const
  {
    auto &tuner = kttHelper.GetTuner();
    // We need to let the rest of the kernels finish, while we won't allow anyone to start a new
    // kernel (this is done by locking the Tuner in Execute method).
    tuner.Synchronize();
    // Now, there are no kernels at the GPU and we can start tuning
    tuner.TuneIteration(kernelId, {});
    tuner.Synchronize();// tmp solution to make the call blocking
    // TODO run should be blocking while tuning -> need change in the KernelLauncher
  }

  /**
   * Runs the kernel with the best known configuration.
   * The call is non-blocking.
   *
   * TODO Second argument is temporary, will be removed once we can acquire the configuration via
   * other means.
   */
  void RunBestConfiguration(ktt::KernelId kernelId, const ktt::KernelConfiguration &TMP = {}) const
  {
    auto &tuner = kttHelper.GetTuner();
    // TODO GetBestConfiguration can be used once the KTT is able to synchronize
    // the best configuration from multiple KTT instances, or loads the best
    // configuration from previous runs
    // auto bestConfig = GetBestConfiguration(kernelId);
    auto bestConfig = TMP;
    tuner.Run(kernelId, bestConfig, {});
  }

protected:
  /**
   * Registers this strategy to the AlgorithmManager.
   * This method is called automatically when the successor class successfully initializes.
   */
  void Register()
  {
    AlgorithmManager::Get().Register(*this);
    isRegistered = true;
  }

  /**
   * Cleans up the KTT ids used by this strategy instance and resets the instance to a default
   * state.
   */
  void Cleanup()
  {
    for (auto &sharedTracker : idTrackers) {
      auto definitionId = sharedTracker->definitionId;
      sharedTracker.reset();
      kttHelper.CleanupIdTracker(definitionId);
    }
    idTrackers.clear();
    definitionIds.clear();
    kernelIds.clear();
    tune = false;
    // FIXME needs to unregister aswell!!!
    isRegistered = false;
  }

  /**
   * Adds KTT's kernel definition.
   * Id of the added kernel definition can be retrieved by using method GetDefinitionId().
   */
  void AddKernelDefinition(const std::string &kernelName,
    const std::string &sourceFile,
    const ktt::DimensionVector &gridDimensions,
    const std::vector<std::string> &templateArgs = {})
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

  /**
   * Adds KTT's kernel.
   * Id of the added kernel can be retrieved by using method GetKernelId().
   */
  void AddKernel(const std::string &name, ktt::KernelDefinitionId definitionId)
  {
    auto kernelId = kttHelper.GetTuner().CreateSimpleKernel(
      name + "_" + std::to_string(strategyId), definitionId);

    if (kernelId == ktt::InvalidKernelId) {
      throw std::invalid_argument("Kernel id could not be created.");
    }

    kernelIds.push_back(kernelId);
    idTrackers.at(GetIndex(definitionId))->kernelIds.push_back(kernelId);
  }

  /**
   * Calls ktt::Tuner::SetArguments method.
   * At the same time registers the ids into an automatic clean up routine. The ids are removed from
   * KTT when they are no longer needed.
   */
  void SetArguments(ktt::KernelDefinitionId id, const std::vector<ktt::ArgumentId> argumentIds)
  {
    kttHelper.GetTuner().SetArguments(id, argumentIds);
    auto &tmp = idTrackers.at(GetIndex(id))->argumentIds;
    tmp.insert(tmp.end(), argumentIds.begin(), argumentIds.end());
  }

  ktt::KernelDefinitionId GetDefinitionId(size_t idx = 0) const { return definitionIds.at(idx); }
  ktt::KernelId GetKernelId(size_t idx = 0) const { return kernelIds.at(idx); }

  utils::KTTHelper &kttHelper;

private:
  std::vector<ktt::KernelDefinitionId> definitionIds;
  std::vector<ktt::KernelId> kernelIds;

  // Tracker of used ids, which allows for automatic cleanup after strategy's destruction
  std::vector<std::shared_ptr<utils::KTTIdTracker>> idTrackers;

  size_t GetIndex(ktt::KernelDefinitionId id) const
  {
    return static_cast<size_t>(std::distance(
      definitionIds.begin(), std::find(definitionIds.begin(), definitionIds.end(), id)));
  }

  // FIXME maybe needs to be new before every InitImpl
  // KTT needs different names for each kernel, this id serves as a simple unique identifier
  const size_t strategyId;

  bool tune;
  bool isRegistered;

  /**
   * Generates new internal id that allows KTT to distinguish different instances of the same
   * strategy.
   */
  static size_t GetNewStrategyId()
  {
    static std::mutex mutex;
    static size_t strategyCounter = 1;
    std::lock_guard<std::mutex> lck(mutex);
    return strategyCounter++;
  }
};

}// namespace umpalumpa::algorithm
