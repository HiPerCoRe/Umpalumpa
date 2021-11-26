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
    if (isRegistered) {
      AlgorithmManager::Get().Unregister(*this);
      Cleanup();
    }
  }

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
   * Return the best known tuning configuration saved in the KTT tuner.
   */
  ktt::KernelConfiguration GetBestConfiguration() const
  {
    return kttHelper.GetTuner().GetBestConfiguration(_kernelId);
  }

  /**
   * Sets a flag that controls whether this strategy should be tuned or not.
   */
  void SetTuning(bool val) { tune = val; }

  /**
   * Returns a flag that that controls whether this strategy should be tuned or not.
   */
  bool ShouldTune() { return tune; }

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
   * Cleans up the KTT ids used by this strategy instance.
   */
  void Cleanup()
  {
    // FIXME TMP remove when Ids in TunableStrategy change to vectors
    std::vector<ktt::KernelId> kernelIds;
    std::vector<ktt::KernelDefinitionId> definitionIds;
    if (_kernelId != ktt::InvalidKernelId) { kernelIds.push_back(_kernelId); }
    if (_definitionId != ktt::InvalidKernelDefinitionId) { definitionIds.push_back(_definitionId); }
    // ^ won't be here
    AlgorithmManager::Get().GetGarbageCollector().CleanupIds(kttHelper, kernelIds, definitionIds);
  }

  // NOTE this method purposefully no longer returns the id, so that people don't use it as a
  // getter, which would be very inefficient and it might cause some troubles during clean up.
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

    // TODO check that id is valid? throw otherwise?
    _definitionId = id;
    AlgorithmManager::Get().GetGarbageCollector().RegisterKernelDefinitionId(id, kttHelper);
  }

  /**
   * Adds KTT's kernel.
   * Id of the added kernel can be retrieved by using method GetKernelId().
   */
  void AddSimpleKernel(const std::string &name, ktt::KernelDefinitionId defId)
  {
    // TODO check that id is valid? throw otherwise?
    _kernelId = kttHelper.GetTuner().CreateSimpleKernel(name + std::to_string(strategyId), defId);
  }

  ktt::KernelDefinitionId GetDefinitionId() const { return _definitionId; }
  ktt::KernelId GetKernelId() const { return _kernelId; }

  utils::KTTHelper &kttHelper;

private:
  // NOTE these might need change to vectors
  // tmp name until changes to vector
  ktt::KernelId _kernelId = ktt::InvalidKernelId;
  ktt::KernelDefinitionId _definitionId = ktt::InvalidKernelDefinitionId;

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
