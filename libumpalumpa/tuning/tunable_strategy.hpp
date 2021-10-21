#pragma once
#include <libumpalumpa/tuning/ktt_helper.hpp>
#include <libumpalumpa/tuning/algorithm_manager.hpp>

namespace umpalumpa::algorithm {

class TunableStrategy
{
  ktt::KernelId kernel;// NOTE this might need change to a vector
  utils::KTTHelper *kttHelper;

public:
  virtual ~TunableStrategy()
  {
    AlgorithmManager::Get().Unregister(this);
    // kttHelper->GetTuner().RemoveKernel(kernel);// FIXME uncomment once the strategies are
    // reworked
  }

  virtual size_t GetHash() const = 0;
  virtual bool IsSimilar(const TunableStrategy &other) const = 0;// Leads to a use of dynamic_cast
  bool IsEquivalent(const TunableStrategy &other) const { return GetHash() == other.GetHash(); }
  std::string GetFullName() const { return typeid(*this).name(); }
  ktt::KernelConfiguration GetBestConfiguration() const
  {
    return kttHelper->GetTuner().GetBestConfiguration(kernel);
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

  void SetKernelId(ktt::KernelId id) { kernel = id; }
};

}// namespace umpalumpa::algorithm
