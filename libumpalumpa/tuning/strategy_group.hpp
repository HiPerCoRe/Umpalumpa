#pragma once
#include <memory>
#include <vector>
#include <limits>
#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/tuning/ktt_strategy_base.hpp>

namespace umpalumpa::tuning {

/**
 * Leader strategy that has properties of TunableStrategy (can be compared using methods: IsEqualTo
 * and IsSimilarTo) and additional properties specific for Leader strategy.
 *
 * Leader strategies are not inteded as strategies that can be initialized and executed!
 */
struct Leader : virtual public detail::TunableStrategyInterface
{
  virtual void SetBestConfigurations(const std::vector<ktt::KernelConfiguration> &configs)
  {
    bestConfigs = configs;
    bestConfigTimes.resize(
      bestConfigs.size(), std::numeric_limits<ktt::Nanoseconds>::max());// FIXME tmp
  }

  virtual void SetBestConfiguration(size_t kernelIndex, const ktt::KernelConfiguration &conf)
  {
    bestConfigs.at(kernelIndex) = conf;
  }

  // FIXME tmp
  virtual void SetBestConfigTime(size_t kernelIndex, ktt::Nanoseconds time)
  {
    bestConfigTimes.at(kernelIndex) = time;
  }

  // FIXME tmp
  virtual ktt::Nanoseconds GetBestConfigTime(size_t kernelIndex) const
  {
    return bestConfigTimes.at(kernelIndex);
  }
  // TODO add methods needed by the Leader class

protected:
  std::vector<ktt::KernelConfiguration> bestConfigs;
  std::vector<ktt::Nanoseconds> bestConfigTimes;// FIXME tmp
};

/**
 * This class groups togerther strategies that are either equal or similar. Each StrategyGroup has
 * one Leader strategy which decides what strategies belong into the StrategyGroup (and whether the
 * strategies are equal or similar). Leader strategy is never removed from the StrategyGroup.
 */
struct StrategyGroup
{

  /**
   * Creates a StrategyGroup with a Leader strategy created out of the provided strategy.
   */
  StrategyGroup(const TunableStrategy &strat) : leader(strat.CreateLeader()) {}

  std::unique_ptr<Leader> leader;
  std::vector<TunableStrategy *> strategies;

  /**
   * Creates a Leader strategy out of the provided strategy. Leader strategies are used to lead
   * StrategyGroups. They provide methods for checking equality and similarity.
   *
   * Leader strategies are not inteded as strategies that can be initialized and executed.
   */
  template<typename Strategy, typename Algorithm>
  static std::unique_ptr<Leader> CreateLeader(const Strategy &s, const Algorithm &a)
  {
    using O = typename Algorithm::OutputData;
    using I = typename Algorithm::InputData;
    using S = typename Algorithm::Settings;
    static_assert(std::is_base_of<BasicAlgorithm<O, I, S>, Algorithm>::value);
    static_assert(std::is_base_of<KTTStrategyBase<O, I, S>, Strategy>::value);
    return std::make_unique<InternalLeader<Strategy>>(s, a);
  }

private:
  /**
   * Leader strategies are used to lead StrategyGroups. They provide methods for checking equality
   * and similarity even without associated BasicAlgorithm class (Leader strategy stores copies of
   * all the needed information).
   * TODO They store additional information (best configurations, whether we still need tuning,
   * etc.) about the StrategyGroup they are leading.
   *
   * Leader strategies are not inteded as strategies that can be initialized and executed!
   */
  template<typename S>
  struct InternalLeader
    : public S
    , public Leader
  {
    // Types (+ compile time check)
    using typename S::StrategyOutput;
    using typename S::StrategyInput;
    using typename S::StrategySettings;

    static_assert(
      std::is_base_of<KTTStrategyBase<StrategyOutput, StrategyInput, StrategySettings>, S>::value);

    /**
     * Returns OutputData stored in the Leader strategy.
     */
    const StrategyOutput &GetOutputRef() const override { return o; }

    /**
     * Returns InputData stored in the Leader strategy.
     */
    const StrategyInput &GetInputRef() const override { return i; }

    /**
     * Returns Settings stored in the Leader strategy.
     */
    const StrategySettings &GetSettings() const override { return s; }

    /**
     * Creates a Leader strategy from the provided strategy and algorithm. Copies necessary
     * information from the provided algorithm.
     */
    InternalLeader(const S &orig,
      const BasicAlgorithm<StrategyOutput, StrategyInput, StrategySettings> &a)
      : S(a), op(orig.GetOutputRef().CopyWithoutData()), ip(orig.GetInputRef().CopyWithoutData()),
        o(op), i(ip), s(orig.GetSettings())
    {}

    ~InternalLeader()
    {
      // TODO save config I guess
    }

    const std::vector<ktt::KernelConfiguration> &GetBestConfigurations() const override
    {
      return bestConfigs;
    }

  private:
    using OutputPayloads = typename StrategyOutput::PayloadCollection;
    using InputPayloads = typename StrategyInput::PayloadCollection;
    OutputPayloads op;
    InputPayloads ip;

    StrategyOutput o;
    StrategyInput i;
    StrategySettings s;
  };
};

}// namespace umpalumpa::tuning

