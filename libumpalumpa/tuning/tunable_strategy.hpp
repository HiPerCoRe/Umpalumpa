#pragma once
#include <libumpalumpa/tuning/ktt_helper.hpp>
#include <libumpalumpa/tuning/algorithm_manager.hpp>

namespace umpalumpa::algorithm {

class TunableStrategy
{
public:
  TunableStrategy() : strategyId(GetNewStrategyId()) {}
  virtual ~TunableStrategy()
  {
    AlgorithmManager::Get().Unregister(this);
    if (kttHelper != nullptr) {
      kttHelper->GetTuner().RemoveKernel(kernelId);
    }// FIXME not working properly can be leak or something
  }

  virtual size_t GetHash() const = 0;
  virtual bool IsSimilar(const TunableStrategy &other) const = 0;// Leads to a use of dynamic_cast
  bool IsEquivalent(const TunableStrategy &other) const { return GetHash() == other.GetHash(); }
  std::string GetFullName() const { return typeid(*this).name(); }
  ktt::KernelConfiguration GetBestConfiguration() const
  {
    return kttHelper->GetTuner().GetBestConfiguration(kernelId);
  }

protected:
  // NOTE Cant be called in constructor because it needs GetHash method to work properly
  /**
   * Needs to be called only once in the Init method of a successor strategy and GetHash method has
   * to return valid output.
   */
  void Init(utils::KTTHelper &helper)
  {
    kttHelper = &helper;
    AlgorithmManager::Get().Register(this);
  }

  ktt::KernelId kernelId;// NOTE this might need change to a vector
  ktt::KernelDefinitionId definitionId;// NOTE this might need change to a vector

  // KTT needs different names for each kernel, this id serves as a simple unique identifier
  const size_t strategyId;

private:
  utils::KTTHelper *kttHelper;

  static size_t GetNewStrategyId()
  {
    static std::mutex mutex;
    static size_t strategyCounter = 1;
    std::lock_guard<std::mutex> lck(mutex);
    return strategyCounter++;
  }
};

}// namespace umpalumpa::algorithm
