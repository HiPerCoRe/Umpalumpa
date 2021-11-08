#pragma once
#include <libumpalumpa/tuning/ktt_helper.hpp>
#include <libumpalumpa/tuning/algorithm_manager.hpp>

namespace umpalumpa::algorithm {

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
    return kttHelper.GetTuner().GetBestConfiguration(kernelId);
  }

  /**
   * Sets a flag that controls whether this strategy should be tuned or not.
   */
  void SetTuning(bool val) { tune = val; }

  /**
   * Returns a flag that that controls whether this strategy shoudl be tuned or not.
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
    if (kernelId != ktt::InvalidKernelId) { kernelIds.push_back(kernelId); }
    if (definitionId != ktt::InvalidKernelDefinitionId) { definitionIds.push_back(definitionId); }
    // ^ won't be here
    AlgorithmManager::Get().CleanupIds(kttHelper, kernelIds, definitionIds);
  }

  /**
   * Returns the KTT's kernel definition Id. If the specified kernel definition already exists,
   * returns the existing Id; otherwise, creates a new one.
   */
  ktt::KernelDefinitionId GetKernelDefinitionId(const std::string &kernelName,
    const std::string &sourceFile,
    const ktt::DimensionVector &gridDimensions,
    const std::vector<std::string> &templateArgs = {})
  {
    return AlgorithmManager::Get().GetKernelDefinitionId(
      kttHelper, kernelName, sourceFile, gridDimensions, templateArgs);
  }

  // NOTE these might need change to vectors
  ktt::KernelId kernelId = ktt::InvalidKernelId;
  ktt::KernelDefinitionId definitionId = ktt::InvalidKernelDefinitionId;

  utils::KTTHelper &kttHelper;

  // KTT needs different names for each kernel, this id serves as a simple unique identifier
  const size_t strategyId;

private:
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
