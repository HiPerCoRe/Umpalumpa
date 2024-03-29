#pragma once
#include <libumpalumpa/tuning/ktt_helper.hpp>
#include <libumpalumpa/tuning/strategy_manager.hpp>
#include <libumpalumpa/tuning/tuning_approach.hpp>

namespace umpalumpa::tuning {

// Forward declarations
struct Leader;

namespace detail {
  /**
   * Interface of important methods that need to be shared between TunableStrategy and Leader
   * strategy. It is used for virtual inheritance of TunableStrategy methods into the Leader. We
   * can't use virtual inheritance without this interface because TunableStrategy has parametrized
   * constructor.
   */
  struct TunableStrategyInterface
  {
    virtual std::unique_ptr<Leader> CreateLeader() const = 0;
    virtual size_t GetHash() const = 0;
    virtual std::string GetFullName() const = 0;
    virtual std::string GetUniqueName() const = 0;
    virtual bool IsSimilarTo(const TunableStrategy &ref) const = 0;
    virtual bool IsEqualTo(const TunableStrategy &ref) const = 0;
    virtual std::vector<ktt::KernelConfiguration> GetDefaultConfigurations() const = 0;
    virtual const std::vector<ktt::KernelConfiguration> &GetBestConfigurations() const = 0;
    virtual ~TunableStrategyInterface() = default;
  };
}// namespace detail

/**
 * Base class for every strategy that uses KTT for tuning.
 * Having this class as a predecessor automates many tasks tied to the tuning process.
 */
class TunableStrategy : virtual public detail::TunableStrategyInterface
{
public:
  /**
   * Creates and initializes TunableStrategy.
   */
  TunableStrategy(KTTHelper &helper);

  /**
   * Destroys the TunableStrategy. Cleans up all the resources (KTT ids) tied to this instance.
   */
  virtual ~TunableStrategy();

  /**
   * Creates a Leader strategy out of this strategy.
   * In order to get the correct type of the strategy, this method needs to be overriden by the
   * successor classes it the following way:
   *
   * return tuning::StrategyGroup::CreateLeader(*this, op);
   */
  std::unique_ptr<Leader> CreateLeader() const override = 0;

  /**
   * Loads tuning data specific for the strategy which calls this method.
   *
   * TODO not satisfied how this works, but can't find anything better now...
   * should be done more automatic
   */
  virtual StrategyGroup LoadTuningData() const = 0;

  /**
   * Returns hash of this strategy. This method needs to be overriden by successor strategy because
   * the hash is computed using operation specific data and settings.
   */
  size_t GetHash() const override = 0;

  /**
   * Each successor strategy defines similarity by its own rules.
   * When we have two similar strategies, we can reuse tuning parameters of one strategy when
   * executing the other one.
   */
  bool IsSimilarTo(const TunableStrategy &ref) const override = 0;

  /**
   * Two strategies are equal when their hashes are the same.
   * When we have two equal strategies, we can use both for tuning.
   */
  bool IsEqualTo(const TunableStrategy &ref) const override;

  /**
   * Returns the full name of the strategy type (including namespaces).
   */
  std::string GetFullName() const override;

  /**
   * Returns unique name of the strategy. Creation of the name should be dependent on the
   * OutputData, InputData, Settings.
   *
   * Name returned by this method should be used as filename of the tuning data.
   */
  std::string GetUniqueName() const override = 0;

  /**
   * Returns default kernel configurations that will be used if there is no other tuned
   * configuration.
   *
   * Each strategy has to override this method and return configurations that are relevant for this
   * strategy.
   */
  std::vector<ktt::KernelConfiguration> GetDefaultConfigurations() const override = 0;
  /**
   * Returns the best known tuning configuration of the specified kernel saved in the KTT tuner.
   */
  virtual ktt::KernelConfiguration GetBestConfiguration(ktt::KernelId kernelId) const;

  /**
   * Returns the best known tuning configurations for all kernels that are part of this strategy.
   */
  const std::vector<ktt::KernelConfiguration> &GetBestConfigurations() const override;

  void AssignLeader(Leader *l) { groupLeader = l; }

  /**
   * Sets a TuningApproach which controls how the strategy should be tuned.
   */
  void SetTuningApproach(TuningApproach approach) { tuningApproach = approach; }

  /**
   * Sets tuning for a specified kernel.
   * This settings has effect only when TuningApproach::kSelectedKernels is set.
   */
  void SetTuningFor(ktt::KernelId kernelId, bool val)
  {
    SetTuningForIdx(GetKernelIndex(kernelId), val);
  }

  /**
   * Sets tuning for a kernel at specified index.
   * This settings has effect only when TuningApproach::kSelectedKernels is set.
   */
  void SetTuningForIdx(size_t idx, bool val) { kernelIds.at(idx).tune = val; }

  /**
   * Allows tuning of the strategy group the strategy is in.
   */
  void AllowTuningStrategyGroup() { canTuneStrategyGroup = true; }

  /**
   * Decides, based on the TuningApproach, whether the provided kernel should be tuned or not.
   */
  bool ShouldBeTuned(ktt::KernelId kernelId) const;

  bool CanTune(ktt::KernelId kernelId) const
  {
    return canTuneStrategyGroup && ShouldBeTuned(kernelId) && HasAnyTuningParameter(kernelId);
  }

  void SetKttLogging(bool val) { kttLoggingOff = !val; }

  void WaitBeforeDestruction(ktt::ComputeActionId actionId) { actionIds.push_back(actionId); }
  void WaitForKernelsToFinish() const;

protected:
  /**
   * Executes the specified kernel. Internally decides whether the strategy will be tuned or not.
   */
  void ExecuteKernel(ktt::KernelId kernelId);

  // Can be moved to private
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
  ktt::KernelResult RunTuning(ktt::KernelId kernelId) const;

  // Can be moved to private
  /**
   * Runs the kernel with the best known configuration.
   * The call is non-blocking.
   */
  void RunBestConfiguration(ktt::KernelId kernelId) const;

  /**
   * If the tuning results are better, it saves them as the new best configuration of the specified
   * kernel.
   */
  void SaveTuningToLeader(ktt::KernelId kernelId, const ktt::KernelResult &tuningResults);

  /**
   * Registers this strategy to the OperationManager.
   * This method is called automatically when the successor class successfully initializes.
   */
  void Register();

  /**
   * Cleans up the KTT ids used by this strategy instance and resets the instance to a default
   * state.
   */
  void Cleanup();

  /**
   * Adds KTT's kernel definition.
   * Id of the added kernel definition can be retrieved by using method GetDefinitionId().
   */
  void AddKernelDefinition(const std::string &kernelName,
    const std::string &sourceFile,
    const ktt::DimensionVector &gridDimensions,
    const std::vector<std::string> &templateArgs = {});

  /**
   * Adds KTT's kernel.
   * Id of the added kernel can be retrieved by using method GetKernelId().
   */
  void AddKernel(const std::string &name, ktt::KernelDefinitionId definitionId);

  /**
   * Calls ktt::Tuner::SetArguments method.
   * At the same time registers the ids into an automatic clean up routine. The ids are removed from
   * KTT when they are no longer needed.
   */
  void SetArguments(ktt::KernelDefinitionId id, const std::vector<ktt::ArgumentId> &argumentIds);

  /**
   * Returns ktt::KernelDefinitionId added by calling method AddKernelDefinition in the order of
   * their addition.
   */
  ktt::KernelDefinitionId GetDefinitionId(size_t idx = 0) const { return definitionIds.at(idx); }

  /**
   * Returns ktt::KernelId added by calling method AddKernel in the order of
   * their addition.
   */
  ktt::KernelId GetKernelId(size_t idx = 0) const { return kernelIds.at(idx).id; }

  bool HasAnyTuningParameter(ktt::KernelId kernelId) const
  {
    return !GetDefaultConfigurations().at(GetKernelIndex(kernelId)).GetPairs().empty();
  }

private:
  /**
   * Returns an internal index of the specified ktt::KernelDefinitionId.
   */
  size_t GetDefinitionIndex(ktt::KernelDefinitionId id) const;

  /**
   * Returns an internal index of the specified ktt::KernelId.
   */
  size_t GetKernelIndex(ktt::KernelId id) const;

  /**
   * Generates new internal id that allows KTT to distinguish different instances of the same
   * strategy.
   */
  static size_t GetNewStrategyId();

protected:
  KTTHelper &kttHelper;

private:
  struct KernelInfo
  {
    ktt::KernelId id = ktt::InvalidKernelId;
    bool tune = true;
  };

  std::vector<ktt::KernelDefinitionId> definitionIds;
  std::vector<KernelInfo> kernelIds;
  // Tracker of used ids, which allows for automatic cleanup after strategy's destruction
  std::vector<std::shared_ptr<KTTIdTracker>> idTrackers;
  std::vector<ktt::ComputeActionId> actionIds;

  Leader *groupLeader = nullptr;
  TuningApproach tuningApproach;
  // the strategy is equal to a Leader of a StrategyGroup and therefore is allowed to be tuned
  bool canTuneStrategyGroup;

  bool isRegistered;
  bool kttLoggingOff = true;

  // FIXME maybe needs to be new before every InitImpl
  // KTT needs different names for each kernel, this id serves as a simple unique identifier
  const size_t strategyId;
};

}// namespace umpalumpa::tuning
