#include <gtest/gtest.h>
#include <libumpalumpa/tuning/algorithm_manager.hpp>

#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include "libumpalumpa/data/logical_desriptor.hpp"
#include "libumpalumpa/tuning/ktt_strategy_base.hpp"
#include "libumpalumpa/tuning/tunable_strategy.hpp"

using namespace umpalumpa;
using namespace umpalumpa::data;
using namespace umpalumpa::algorithm;

template<typename T = Payload<LogicalDescriptor>> struct DataWrapper : public PayloadWrapper<T>
{
  DataWrapper(std::tuple<T> &&t) : PayloadWrapper<T>(std::move(t)) {}
  DataWrapper(T d) : PayloadWrapper<T>(std::move(d)) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
};

struct Settings
{
  // Empty
};

// TODO ideally mock object
// TODO TestStrategy should be mocked!!! really needed to properly test this class
class TestAlgorithm_CUDA
  : public BasicAlgorithm<DataWrapper<>, DataWrapper<>, Settings>
  , public KTT_Base
{
public:
  TestAlgorithm_CUDA(bool stratInitReturn = true) : KTT_Base(0), stratInitReturn(stratInitReturn) {}

  using KTTStrategy = algorithm::KTTStrategyBase<OutputData, InputData, Settings>;

  void Synchronize() override {}

  struct TestStrategy : public KTTStrategy
  {
    TestStrategy(const BasicAlgorithm &algo, bool initReturn)
      : KTTStrategy(algo), initReturn(initReturn)
    {}

    bool InitImpl() override { return initReturn; }

    bool Execute(const OutputData &, const InputData &) override { return true; }

    std::string GetName() const override { return "TestStrategy"; }
    size_t GetHash() const override { return 0; }
    bool IsSimilarTo(const TunableStrategy &) const override { return false; }

  protected:
    bool initReturn;
  };

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override
  {
    std::vector<std::unique_ptr<Strategy>> vec;
    vec.emplace_back(std::make_unique<TestStrategy>(*this, stratInitReturn));
    return vec;
  }

  bool IsValid(const OutputData &, const InputData &, const Settings &) const override
  {
    return true;
  }

  bool stratInitReturn;
};

// NOTE AlgorithmManager is a singleton and therefore has an global state. It needs to be reset
// before each test SetUp/TearDown of gTest fixtures would help with that... TODO rework to use
// fixtures

TEST(AlgorithmManagerTests, strategy_registered_automatically_when_init_succeeds)
{
  AlgorithmManager::Get().Reset();
  Settings settings;

  Size size(42, 1, 1, 1);

  LogicalDescriptor ld(size);
  PhysicalDescriptor pd(0, DataType::kFloat);
  auto inP = TestAlgorithm_CUDA::InputData(Payload(nullptr, ld, pd, "Input data"));
  auto outP = TestAlgorithm_CUDA::OutputData(Payload(nullptr, ld, pd, "Output data"));

  TestAlgorithm_CUDA algo;

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(algo.Init(outP, inP, settings));
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
}

TEST(AlgorithmManagerTests, strategy_not_registered_when_init_failed)
{
  AlgorithmManager::Get().Reset();
  Settings settings;

  Size size(42, 1, 1, 1);

  LogicalDescriptor ld(size);
  PhysicalDescriptor pd(0, DataType::kFloat);
  auto inP = TestAlgorithm_CUDA::InputData(Payload(nullptr, ld, pd, "Input data"));
  auto outP = TestAlgorithm_CUDA::OutputData(Payload(nullptr, ld, pd, "Output data"));

  TestAlgorithm_CUDA algo(false);

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  ASSERT_FALSE(algo.Init(outP, inP, settings));
  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
}

TEST(AlgorithmManagerTests, strategy_unregistered_automatically_when_destroyed)
{
  AlgorithmManager::Get().Reset();
  Settings settings;

  Size size(42, 1, 1, 1);

  LogicalDescriptor ld(size);
  PhysicalDescriptor pd(0, DataType::kFloat);
  auto inP = TestAlgorithm_CUDA::InputData(Payload(nullptr, ld, pd, "Input data"));
  auto outP = TestAlgorithm_CUDA::OutputData(Payload(nullptr, ld, pd, "Output data"));

  {
    TestAlgorithm_CUDA algo;

    ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
    ASSERT_TRUE(algo.Init(outP, inP, settings));
    ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
  }// the algorithm and the strategy are destroyed

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
}

// test for automatic vector removal when empty

TEST(AlgorithmManagerTests, strategy_already_registered_cant_register_again)
{
  // NOTE This test uses classes and their methods in an unusual (unintended) way to force this
  // situation
  AlgorithmManager::Get().Reset();
  Settings settings;

  Size size(42, 1, 1, 1);

  LogicalDescriptor ld(size);
  PhysicalDescriptor pd(0, DataType::kFloat);
  auto inP = TestAlgorithm_CUDA::InputData(Payload(nullptr, ld, pd, "Input data"));
  auto outP = TestAlgorithm_CUDA::OutputData(Payload(nullptr, ld, pd, "Output data"));

  TestAlgorithm_CUDA algo;

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(algo.Init(outP, inP, settings));
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
  // Now we have 1 strategy registered and now we try to insert again the same strategy
  // The following line is implementation specific and may need changes when AlgorithmManager
  // internals change
  auto *stratPtr = AlgorithmManager::Get().GetRegisteredStrategies().at(0).at(0);
  AlgorithmManager::Get().Register(*stratPtr);// Try to add the same strategy again
  // Nothing should change
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
}

TEST(AlgorithmManagerTests, not_registered_strategy_cant_be_unregistered)
{
  AlgorithmManager::Get().Reset();
  // No way to reliably test this situation because we can't access strategy that is not registered
  // SHOULD DO NOTHING
}

TEST(AlgorithmManagerTests,
  GetBestConfiguration_returns_empty_configuration_when_passed_invalid_hash)
{
  AlgorithmManager::Get().Reset();
  // AlgorithmManager is empty therefore every hash is invalid one
  // (this might not be true when DB is added)(DB should be mocked in tests, so it will be true)
  ASSERT_TRUE(AlgorithmManager::Get().GetBestConfiguration(0).GetPairs().empty());
}

TEST(AlgorithmManagerTests,
  strategy_registered_GetBestConfiguration_returns_best_config_when_passed_valid_hash)
{
  AlgorithmManager::Get().Reset();
  Settings settings;

  Size size(42, 1, 1, 1);

  LogicalDescriptor ld(size);
  PhysicalDescriptor pd(0, DataType::kFloat);
  auto inP = TestAlgorithm_CUDA::InputData(Payload(nullptr, ld, pd, "Input data"));
  auto outP = TestAlgorithm_CUDA::OutputData(Payload(nullptr, ld, pd, "Output data"));

  TestAlgorithm_CUDA algo;

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(algo.Init(outP, inP, settings));
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
  // We need to execute the algorithm, so that KTT can generate some config
  // TODO either properly define TestAlgorithm_CUDA to use KTT or generate the config in some other
  // way
  ASSERT_TRUE(algo.Execute(outP, inP));

  auto &strats = AlgorithmManager::Get().GetRegisteredStrategies();
  auto bestConfig = AlgorithmManager::Get().GetBestConfiguration(strats.at(0).at(0)->GetHash());
  // TODO some assert
}

TEST(AlgorithmManagerTests,
  strategy_not_registered_GetBestConfiguration_returns_best_config_when_passed_valid_hash)
{
  AlgorithmManager::Get().Reset();
  // TODO when DB is added (DB should be mocked here!!!)
}

TEST(AlgorithmManagerTests, multiple_strategies_registered_at_once)
{
  AlgorithmManager::Get().Reset();
  // TODO
}

TEST(AlgorithmManagerTests, indexing_in_strategies_using_hash_TODONAME)
{
  AlgorithmManager::Get().Reset();
  // TODO
}

