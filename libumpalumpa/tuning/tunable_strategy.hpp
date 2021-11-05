#pragma once
#include <libumpalumpa/tuning/ktt_helper.hpp>
#include <libumpalumpa/tuning/algorithm_manager.hpp>

namespace umpalumpa::algorithm {

class TunableStrategy
{
public:
  TunableStrategy(utils::KTTHelper &helper)
    : kttHelper(helper), strategyId(GetNewStrategyId()), tune(false), isRegistered(false)
  {}

  virtual ~TunableStrategy()
  {
    if (isRegistered) {
      AlgorithmManager::Get().Unregister(*this);
      Cleanup();
    }
  }

  virtual size_t GetHash() const = 0;

  virtual bool IsSimilarTo(const TunableStrategy &ref) const = 0;

  bool IsEqualTo(const TunableStrategy &ref) const { return GetHash() == ref.GetHash(); }

  std::string GetFullName() const { return typeid(*this).name(); }

  ktt::KernelConfiguration GetBestConfiguration() const
  {
    return kttHelper.GetTuner().GetBestConfiguration(kernelId);
  }

  void SetTuning(bool val) { tune = val; }

  bool ShouldTune() { return tune; }

protected:
  /**
   * This method is called automatically when the successor class successfully initializes.
   */
  void Register()
  {
    AlgorithmManager::Get().Register(*this);
    isRegistered = true;
  }

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

  static size_t GetNewStrategyId()
  {
    static std::mutex mutex;
    static size_t strategyCounter = 1;
    std::lock_guard<std::mutex> lck(mutex);
    return strategyCounter++;
  }
};

}// namespace umpalumpa::algorithm
