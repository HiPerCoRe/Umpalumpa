#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <libumpalumpa/tuning/algorithm_manager.hpp>

#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include "libumpalumpa/data/logical_desriptor.hpp"
#include "libumpalumpa/tuning/ktt_strategy_base.hpp"
#include "libumpalumpa/tuning/tunable_strategy.hpp"

using namespace ::testing;
using namespace umpalumpa;
using namespace umpalumpa::data;
using namespace umpalumpa::algorithm;

// PREPARATION OF TEST CLASSES

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

class TestAlgorithm_CUDA
  : public BasicAlgorithm<DataWrapper<>, DataWrapper<>, Settings>
  , public KTT_Base
{
public:
  TestAlgorithm_CUDA() : KTT_Base(0) {}

  using KTTStrategy = algorithm::KTTStrategyBase<OutputData, InputData, Settings>;

  void Synchronize() override {}

  struct MockStrategy : public KTTStrategy
  {
    using KTTStrategy::KTTStrategy;
    std::string GetName() const override { return "MockStrategy"; }

    MOCK_METHOD(bool, InitImpl, (), (override));
    MOCK_METHOD(bool, Execute, (const OutputData &, const InputData &), (override));
    MOCK_METHOD(bool, IsSimilarTo, (const TunableStrategy &), (const, override));
    MOCK_METHOD(size_t, GetHash, (), (const, override));
  };

protected:
  std::vector<std::unique_ptr<Strategy>> GetStrategies() const override
  {
    std::vector<std::unique_ptr<Strategy>> vec;
    vec.emplace_back(std::unique_ptr<MockStrategy>(mockStratPtr));
    return vec;
  }

  bool IsValid(const OutputData &, const InputData &, const Settings &) const override
  {
    return true;
  }

public:
  // Unique ptr is taking care of this
  MockStrategy *mockStratPtr = new MockStrategy(*this);
};

// ACTUAL TESTS ARE STARTING HERE

// NOTE some tests are using knowledge of AlgorithmManager's internal structure
// (that it stores strategies in nested vectors), would be good to somehow fix it
// so that the tests are not implementation dependent

class AlgorithmManagerTests : public Test
{
protected:
  // Constructor is called before every test
  AlgorithmManagerTests()
    : settings(), size(42, 1, 1, 1), ld(size),
      pd(nullptr, 0, DataType::kFloat, ManagedBy::Manually, 0), inP(Payload(ld, pd, "Input data")),
      outP(Payload(ld, pd, "Output data"))
  {
    // NOTE AlgorithmManager is a singleton and therefore has a global state. It needs to be reset
    // before each test.
    AlgorithmManager::Get().Cleanup();
  }

  const Settings settings;
  const Size size;
  const LogicalDescriptor ld;
  const PhysicalDescriptor pd;

  const TestAlgorithm_CUDA::InputData inP;
  const TestAlgorithm_CUDA::OutputData outP;
};

TEST_F(AlgorithmManagerTests, strategy_registered_automatically_when_init_succeeds)
{
  TestAlgorithm_CUDA algo;
  EXPECT_CALL(*algo.mockStratPtr, InitImpl()).WillOnce(Return(true));

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(algo.Init(outP, inP, settings));
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 1u);
}

TEST_F(AlgorithmManagerTests, strategy_not_registered_when_init_failed)
{
  TestAlgorithm_CUDA algo;
  EXPECT_CALL(*algo.mockStratPtr, InitImpl()).WillOnce(Return(false));

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  ASSERT_FALSE(algo.Init(outP, inP, settings));
  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
}

TEST_F(AlgorithmManagerTests, strategy_unregistered_automatically_when_destroyed)
{
  {
    TestAlgorithm_CUDA algo;
    EXPECT_CALL(*algo.mockStratPtr, InitImpl()).WillOnce(Return(true));

    ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
    ASSERT_TRUE(algo.Init(outP, inP, settings));
    ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
    ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 1u);
  }// the algorithm and the strategy are destroyed

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
}

TEST_F(AlgorithmManagerTests, strategy_already_registered_cant_register_again)
{
  TestAlgorithm_CUDA algo;
  EXPECT_CALL(*algo.mockStratPtr, InitImpl()).WillOnce(Return(true));

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(algo.Init(outP, inP, settings));
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
  // Now we have 1 strategy registered and now we try to insert again the same strategy
  // To force this situation we use explicit call to Register method
  AlgorithmManager::Get().Register(*algo.mockStratPtr);// Try to add the same strategy again
  // Nothing should change
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 1u);
}

TEST_F(AlgorithmManagerTests, not_registered_strategy_cant_be_unregistered)
{
  TestAlgorithm_CUDA algo;

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  AlgorithmManager::Get().Unregister(*algo.mockStratPtr);// Should print a warning
  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
}

TEST_F(AlgorithmManagerTests,
  GetBestConfiguration_returns_empty_configuration_when_passed_invalid_hash)
{
  // AlgorithmManager is empty therefore every hash is invalid one
  // When DB is added, it needs to be mocked in tests
  ASSERT_TRUE(AlgorithmManager::Get().GetBestConfiguration(0).GetPairs().empty());
}

TEST_F(AlgorithmManagerTests,
  strategy_registered_GetBestConfiguration_returns_best_config_when_passed_valid_hash)
{
  TestAlgorithm_CUDA algo;
  EXPECT_CALL(*algo.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*algo.mockStratPtr, Execute).WillOnce(Return(true));
  EXPECT_CALL(*algo.mockStratPtr, GetHash()).WillRepeatedly(Return(42));

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(algo.Init(outP, inP, settings));
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 1u);
  // We need to execute the algorithm, so that KTT can generate some config
  // TODO either properly define TestAlgorithm_CUDA to use KTT or generate the config in some other
  // way
  ASSERT_TRUE(algo.Execute(outP, inP));

  auto bestConfig = AlgorithmManager::Get().GetBestConfiguration(algo.mockStratPtr->GetHash());
  // TODO some assert
}

TEST_F(AlgorithmManagerTests,
  strategy_not_registered_GetBestConfiguration_returns_best_config_when_passed_valid_hash)
{
  // TODO when DB is added (DB should be mocked here!!!)
}

TEST_F(AlgorithmManagerTests, multiple_equivalent_strategies_registered_at_once)
{
  TestAlgorithm_CUDA algo1;
  TestAlgorithm_CUDA algo2;
  TestAlgorithm_CUDA algo3;
  EXPECT_CALL(*algo1.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*algo2.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*algo3.mockStratPtr, InitImpl()).WillOnce(Return(true));

  // Same hash == equivalence
  EXPECT_CALL(*algo1.mockStratPtr, GetHash()).WillRepeatedly(Return(42));
  EXPECT_CALL(*algo2.mockStratPtr, GetHash()).WillRepeatedly(Return(42));
  EXPECT_CALL(*algo3.mockStratPtr, GetHash()).WillRepeatedly(Return(42));

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(algo1.Init(outP, inP, settings));
  ASSERT_TRUE(algo2.Init(outP, inP, settings));
  ASSERT_TRUE(algo3.Init(outP, inP, settings));
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 3u);
}

TEST_F(AlgorithmManagerTests, multiple_similar_strategies_registered_at_once)
{
  TestAlgorithm_CUDA algo1;
  TestAlgorithm_CUDA algo2;
  TestAlgorithm_CUDA algo3;
  EXPECT_CALL(*algo1.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*algo2.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*algo3.mockStratPtr, InitImpl()).WillOnce(Return(true));

  // Different hash != equivalence
  EXPECT_CALL(*algo1.mockStratPtr, GetHash()).WillRepeatedly(Return(123));
  EXPECT_CALL(*algo2.mockStratPtr, GetHash()).WillRepeatedly(Return(456));
  EXPECT_CALL(*algo3.mockStratPtr, GetHash()).WillRepeatedly(Return(789));

  // But they are similar
  EXPECT_CALL(*algo1.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));
  EXPECT_CALL(*algo2.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));
  EXPECT_CALL(*algo3.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(algo1.Init(outP, inP, settings));
  ASSERT_TRUE(algo2.Init(outP, inP, settings));
  ASSERT_TRUE(algo3.Init(outP, inP, settings));
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 3u);
}

TEST_F(AlgorithmManagerTests, multiple_different_strategies_registered_at_once)
{
  TestAlgorithm_CUDA algo1;
  TestAlgorithm_CUDA algo2;
  TestAlgorithm_CUDA algo3;
  EXPECT_CALL(*algo1.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*algo2.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*algo3.mockStratPtr, InitImpl()).WillOnce(Return(true));

  // Different hash != equivalence
  EXPECT_CALL(*algo1.mockStratPtr, GetHash()).WillRepeatedly(Return(123));
  EXPECT_CALL(*algo2.mockStratPtr, GetHash()).WillRepeatedly(Return(456));
  EXPECT_CALL(*algo3.mockStratPtr, GetHash()).WillRepeatedly(Return(789));

  // They are not similar
  EXPECT_CALL(*algo1.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));
  EXPECT_CALL(*algo2.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));
  EXPECT_CALL(*algo3.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));

  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(algo1.Init(outP, inP, settings));
  ASSERT_TRUE(algo2.Init(outP, inP, settings));
  ASSERT_TRUE(algo3.Init(outP, inP, settings));
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 3u);
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 1u);
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(1).size(), 1u);
  ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(2).size(), 1u);
}

TEST_F(AlgorithmManagerTests, multiple_similar_strategies_unregistered_correctly)
{
  {
    TestAlgorithm_CUDA algo1;
    {
      TestAlgorithm_CUDA algo2;
      {
        TestAlgorithm_CUDA algo3;

        EXPECT_CALL(*algo1.mockStratPtr, InitImpl()).WillOnce(Return(true));
        EXPECT_CALL(*algo2.mockStratPtr, InitImpl()).WillOnce(Return(true));
        EXPECT_CALL(*algo3.mockStratPtr, InitImpl()).WillOnce(Return(true));

        // Different hash != equivalence
        EXPECT_CALL(*algo1.mockStratPtr, GetHash()).WillRepeatedly(Return(123));
        EXPECT_CALL(*algo2.mockStratPtr, GetHash()).WillRepeatedly(Return(456));
        EXPECT_CALL(*algo3.mockStratPtr, GetHash()).WillRepeatedly(Return(789));

        // But they are similar
        EXPECT_CALL(*algo1.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));
        EXPECT_CALL(*algo2.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));
        EXPECT_CALL(*algo3.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));

        ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
        ASSERT_TRUE(algo1.Init(outP, inP, settings));
        ASSERT_TRUE(algo2.Init(outP, inP, settings));
        ASSERT_TRUE(algo3.Init(outP, inP, settings));
        ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
        ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 3u);
      }// algo3 is destroyed, therefore its strategy is unregistered
      ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
      ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 2u);
    }// algo2 is destroyed, therefore its strategy is unregistered
    ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
    ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 1u);
  }// algo1 is destroyed, therefore its strategy is unregistered
  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
}

TEST_F(AlgorithmManagerTests, multiple_different_strategies_unregistered_correctly)
{
  {
    TestAlgorithm_CUDA algo1;
    {
      TestAlgorithm_CUDA algo2;
      {
        TestAlgorithm_CUDA algo3;

        EXPECT_CALL(*algo1.mockStratPtr, InitImpl()).WillOnce(Return(true));
        EXPECT_CALL(*algo2.mockStratPtr, InitImpl()).WillOnce(Return(true));
        EXPECT_CALL(*algo3.mockStratPtr, InitImpl()).WillOnce(Return(true));

        // Different hash != equivalence
        EXPECT_CALL(*algo1.mockStratPtr, GetHash()).WillRepeatedly(Return(123));
        EXPECT_CALL(*algo2.mockStratPtr, GetHash()).WillRepeatedly(Return(456));
        EXPECT_CALL(*algo3.mockStratPtr, GetHash()).WillRepeatedly(Return(789));

        // They are not similar
        EXPECT_CALL(*algo1.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));
        EXPECT_CALL(*algo2.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));
        EXPECT_CALL(*algo3.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));

        ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
        ASSERT_TRUE(algo1.Init(outP, inP, settings));
        ASSERT_TRUE(algo2.Init(outP, inP, settings));
        ASSERT_TRUE(algo3.Init(outP, inP, settings));
        ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 3u);
        ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 1u);
        ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(1).size(), 1u);
        ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(2).size(), 1u);
      }// algo3 is destroyed, therefore its strategy is unregistered
      ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 2u);
      ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 1u);
      ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(1).size(), 1u);
    }// algo2 is destroyed, therefore its strategy is unregistered
    ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().size(), 1u);
    ASSERT_EQ(AlgorithmManager::Get().GetRegisteredStrategies().at(0).size(), 1u);
  }// algo1 is destroyed, therefore its strategy is unregistered
  ASSERT_TRUE(AlgorithmManager::Get().GetRegisteredStrategies().empty());
}
