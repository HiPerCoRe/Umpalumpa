#pragma once

#include "libumpalumpa/data/logical_desriptor.hpp"
#include "libumpalumpa/tuning/strategy_group.hpp"
#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/ktt_strategy_base.hpp>
#include <libumpalumpa/data/payload_wrapper.hpp>

namespace umpalumpa {
template<typename T = data::Payload<data::LogicalDescriptor>>
struct DataWrapper : public data::PayloadWrapper<T>
{
  DataWrapper(std::tuple<T> &t) : data::PayloadWrapper<T>(t) {}
  DataWrapper(T &data) : data::PayloadWrapper<T>(data) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
  typedef T PayloadType;
};

class Settings
{
};

class WaitingAlgorithm
  : public BasicAlgorithm<DataWrapper<>, DataWrapper<>, Settings>
  , public tuning::KTT_Base
{
public:
  using tuning::KTT_Base::KTT_Base;
  using BasicAlgorithm::Strategy;
  using KTTStrategy = tuning::KTTStrategyBase<OutputData, InputData, Settings>;
  void Synchronize() override { GetHelper().GetTuner().Synchronize(); }

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override;
};

namespace {
  inline static const auto kKernelFile = utils::GetSourceFilePath("tests/tuning/waiting_kernel.cu");

  struct WaitingStrategy : public WaitingAlgorithm::KTTStrategy
  {
    using WaitingAlgorithm::KTTStrategy::KTTStrategy;

    static constexpr auto kWaitingKernel = "waitingKernel";
    size_t GetHash() const override { return 0; }
    std::unique_ptr<tuning::Leader> CreateLeader() const override
    {
      return tuning::StrategyGroup::CreateLeader(*this, alg);
    }
    std::vector<ktt::KernelConfiguration> GetDefaultConfigurations() const override
    {
      return { kttHelper.GetTuner().CreateConfiguration(
        GetKernelId(), { { "MILLISECONDS", static_cast<uint64_t>(5) } }) };
    }
    bool IsEqualTo(const TunableStrategy &ref) const override
    {
      bool equal = true;
      try {
        equal = IsSimilarTo(dynamic_cast<const WaitingStrategy &>(ref));
      } catch (std::bad_cast &) {
        equal = false;
      }
      return equal;
    }
    bool IsSimilarTo(const TunableStrategy &) const override { return true; }
    std::string GetName() const override final { return "WaitingStrategy"; }
    bool InitImpl() override
    {
      AddKernelDefinition(kWaitingKernel, kKernelFile, ktt::DimensionVector{});
      auto definitionId = GetDefinitionId();
      AddKernel(kWaitingKernel, definitionId);
      auto kernelId = GetKernelId();
      auto &tuner = kttHelper.GetTuner();
      tuner.AddParameter(kernelId, "MILLISECONDS", std::vector<uint64_t>{ 2, 4, 6, 8, 10 });
      tuner.SetSearcher(kernelId, std::make_unique<ktt::RandomSearcher>());
      return true;
    }
    bool Execute(const WaitingAlgorithm::OutputData &,
      const WaitingAlgorithm::InputData &) override final
    {
      kttHelper.GetTuner().SetLauncher(GetKernelId(), [this](ktt::ComputeInterface &interface) {
        interface.RunKernelAsync(GetDefinitionId(), interface.GetAllQueues().at(0));
      });
      ExecuteKernel(GetKernelId());
      return true;
    }
  };
}// namespace

std::vector<std::unique_ptr<WaitingAlgorithm::Strategy>> WaitingAlgorithm::GetStrategies() const
{
  std::vector<std::unique_ptr<WaitingAlgorithm::Strategy>> vec;
  vec.emplace_back(std::make_unique<WaitingStrategy>(*this));
  return vec;
}

}// namespace umpalumpa
