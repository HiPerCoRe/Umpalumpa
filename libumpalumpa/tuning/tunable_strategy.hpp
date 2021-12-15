#pragma once
#include <libumpalumpa/tuning/ktt_helper.hpp>
#include <libumpalumpa/tuning/algorithm_manager.hpp>
#include <libumpalumpa/tuning/tuning_approach.hpp>

namespace umpalumpa::algorithm {

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
  TunableStrategy(utils::KTTHelper &helper);

  /**
   * Destroys the TunableStrategy. Cleans up all the resources (KTT ids) tied to this instance.
   */
  virtual ~TunableStrategy();

  /**
   * Creates a Leader strategy out of this strategy.
   * In order to get the correct type of the strategy, this method needs to be overriden by the
   * successor classes it the following way:
   *
   * return algorithm::StrategyGroup::CreateLeader(*this, alg);
   */
  std::unique_ptr<Leader> CreateLeader() const override = 0;

  /**
   * Returns hash of this strategy. This method needs to be overriden by successor strategy because
   * the hash is computed using algorithm specific data and settings.
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
  std::string GetFullName() const;

  /**
   * TODO
   */
  std::vector<ktt::KernelConfiguration> GetDefaultConfigurations() const override = 0;
  /**
   * Returns the best known tuning configuration of the specified kernel saved in the KTT tuner.
   */
  virtual ktt::KernelConfiguration GetBestConfiguration(ktt::KernelId kernelId) const;

  /**
   * TODO
   */
  const std::vector<ktt::KernelConfiguration> &GetBestConfigurations() const override;

  // TODO move to private + Setter
  const Leader *groupLeader = nullptr;

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
    kernelIds.at(GetKernelIndex(kernelId)).tune = val;
  }

  // FIXME find better name
  /**
   * Allows tuning of the strategy group the strategy is in.
   */
  void AllowTuningStrategyGroup() { canTuneStrategyGroup = true; }

  /**
   * Decides, based on the TuningApproach, whether the provided kernel should be tuned or not.
   */
  bool ShouldBeTuned(ktt::KernelId kernelId) const;

protected:
  /**
   * Executes the specified kernel. Internally decides whether the strategy will be tuned or not.
   */
  void ExecuteKernel(ktt::KernelId kernelId) const;

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
  void RunTuning(ktt::KernelId kernelId) const;

  // Can be moved to private
  /**
   * Runs the kernel with the best known configuration.
   * The call is non-blocking.
   */
  void RunBestConfiguration(ktt::KernelId kernelId) const;

  /**
   * Registers this strategy to the AlgorithmManager.
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
  utils::KTTHelper &kttHelper;

private:
  struct KernelInfo
  {
    ktt::KernelId id = ktt::InvalidKernelId;
    bool tune = false;
  };

  std::vector<ktt::KernelDefinitionId> definitionIds;
  std::vector<KernelInfo> kernelIds;
  // Tracker of used ids, which allows for automatic cleanup after strategy's destruction
  std::vector<std::shared_ptr<utils::KTTIdTracker>> idTrackers;

  TuningApproach tuningApproach;
  // the strategy is equal to a Leader of a StrategyGroup and therefore is allowed to be tuned
  bool canTuneStrategyGroup;

  bool isRegistered;

  // FIXME maybe needs to be new before every InitImpl
  // KTT needs different names for each kernel, this id serves as a simple unique identifier
  const size_t strategyId;
};

}// namespace umpalumpa::algorithm
