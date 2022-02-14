#pragma once
#include <memory>
#include <vector>
#include <limits>
#include <fstream>
#include <sstream>
#include <filesystem>
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
  Leader() = default;
  Leader(std::vector<ktt::KernelConfiguration> &&vConf, std::vector<ktt::Nanoseconds> &&vTime)
    : bestConfigs(vConf), bestConfigTimes(vTime)
  {}

  virtual void SetBestConfigurations(const std::vector<ktt::KernelConfiguration> &configs)
  {
    bestConfigs = configs;
    bestConfigTimes.resize(
      bestConfigs.size(), std::numeric_limits<ktt::Nanoseconds>::max());// FIXME tmp
  }

  // FIXME make one more robust method for storing the best config instead of how it is now
  // this will need changes in TunableStrategy aswell

  virtual void SetBestConfiguration(size_t kernelIndex, const ktt::KernelConfiguration &conf)
  {
    bestConfigs.at(kernelIndex) = conf;
  }

  virtual void SetBestConfigTime(size_t kernelIndex, ktt::Nanoseconds time)
  {
    bestConfigTimes.at(kernelIndex) = time;
  }

  virtual const ktt::KernelConfiguration &GetBestConfiguration(size_t kernelIndex) const
  {
    return bestConfigs.at(kernelIndex);
  }

  virtual ktt::Nanoseconds GetBestConfigTime(size_t kernelIndex) const
  {
    return bestConfigTimes.at(kernelIndex);
  }

  virtual void Serialize(std::ostream &out) const = 0;

  // TODO add methods needed by the Leader class
protected:
  std::vector<ktt::KernelConfiguration> bestConfigs;
  std::vector<ktt::Nanoseconds> bestConfigTimes;// FIXME tmp
};

/**
 * This class groups together strategies that are either equal or similar. Each StrategyGroup has
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

  template<typename Strategy, typename Algorithm>
  static StrategyGroup LoadTuningData(const Strategy &s, const Algorithm &a)
  {
    auto filePath = utils::GetTuningDirectory() + s.GetUniqueName();
    std::ifstream inputFile(filePath);
    if (!inputFile) { throw std::logic_error("Could not open file: " + filePath); }

    return StrategyGroup(InternalLeader<Strategy>::Deserialize(a, inputFile));
  }

  void Serialize(std::ostream &out) const { leader->Serialize(out); }

  bool IsEqualTo(const StrategyGroup &ref) const
  {
    const auto &refLeader = *ref.leader;
    return leader->IsEqualTo(dynamic_cast<const TunableStrategy &>(refLeader));
  }

  void Merge(const StrategyGroup &other)
  {
    for (size_t idx = 0; idx < leader->GetBestConfigurations().size(); ++idx) {
      if (other.leader->GetBestConfigTime(idx) < leader->GetBestConfigTime(idx)) {
        // Might just move it, but not sure if it is safe right now
        leader->SetBestConfiguration(idx, other.leader->GetBestConfiguration(idx));
        leader->SetBestConfigTime(idx, other.leader->GetBestConfigTime(idx));
      }
    }
  }

private:
  template<typename S> struct InternalLeader;

  template<typename Strategy>
  StrategyGroup(std::unique_ptr<InternalLeader<Strategy>> &&l) : leader(std::move(l))
  {}

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

    using StratType = S;
    using AlgType = BasicAlgorithm<StrategyOutput, StrategyInput, StrategySettings>;

    std::string GetFullName() const override { return typeid(S).name(); }
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
    InternalLeader(const S &orig, const AlgType &a)
      : S(a), op(orig.GetOutputRef().CopyWithoutData()), ip(orig.GetInputRef().CopyWithoutData()),
        o(op), i(ip), s(orig.GetSettings())
    {
      S::SetUniqueStrategyName();
    }

    ~InternalLeader() {}

    const std::vector<ktt::KernelConfiguration> &GetBestConfigurations() const override
    {
      return bestConfigs;
    }

    void Serialize(std::ostream &out) const override
    {
      o.Serialize(out);
      i.Serialize(out);
      s.Serialize(out);
      auto size = bestConfigs.size();
      out << size << '\n';
      for (size_t idx = 0; idx < size; ++idx) {
        out << bestConfigTimes.at(idx);
        for (const auto &pp : bestConfigs.at(idx).GetPairs()) {
          out << ' ' << pp.HasValueDouble() << ' ' << pp.GetName() << ' ' << pp.GetValue();
        }
        out << '\n';
      }
    }

    static auto Deserialize(const AlgType &a, std::istream &in)
    {
      auto outData = StrategyOutput::Deserialize(in);
      auto inData = StrategyInput::Deserialize(in);
      auto settings = StrategySettings::Deserialize(in);
      std::vector<ktt::KernelConfiguration> vConf;
      std::vector<ktt::Nanoseconds> vTime;

      size_t size;
      in >> size;
      for (size_t idx = 0; idx < size; ++idx) {
        ktt::Nanoseconds time;
        in >> time;
        std::string line;
        std::getline(in, line);
        std::stringstream ss(line + '\n');
        std::vector<ktt::ParameterPair> params;
        while (ss.peek() != '\n') {
          bool isDouble;
          std::string name;
          ss >> isDouble >> name;
          if (isDouble) {
            double val;
            ss >> val;
            params.emplace_back(name, val);
          } else {
            uint64_t val;
            ss >> val;
            params.emplace_back(name, val);
          }
        }
        vConf.emplace_back(params);// ktt::KernelConfiguration isn't move-constructible
        vTime.emplace_back(time);
      }

      return std::make_unique<InternalLeader>(a,
        std::move(outData),
        std::move(inData),
        std::move(settings),
        std::move(vConf),
        std::move(vTime));
    }

  private:
    using OutputPayloads = typename StrategyOutput::PayloadCollection;
    using InputPayloads = typename StrategyInput::PayloadCollection;

  public:
    InternalLeader(const AlgType &a,
      OutputPayloads &&ops,
      InputPayloads &&ips,
      StrategySettings &&ss,
      std::vector<ktt::KernelConfiguration> &&vConf,
      std::vector<ktt::Nanoseconds> &&vTime)
      : S(a), Leader(std::move(vConf), std::move(vTime)), op(std::move(ops)), ip(std::move(ips)),
        o(op), i(ip), s(std::move(ss))
    {
      S::SetUniqueStrategyName();
    }

  private:
    OutputPayloads op;
    InputPayloads ip;

    StrategyOutput o;
    StrategyInput i;
    StrategySettings s;
  };
};

}// namespace umpalumpa::tuning

