#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <libumpalumpa/tuning/strategy_manager.hpp>

#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/operations/basic_operation.hpp>
#include "libumpalumpa/data/logical_desriptor.hpp"
#include "libumpalumpa/tuning/ktt_strategy_base.hpp"
#include "libumpalumpa/tuning/tunable_strategy.hpp"
#include <libumpalumpa/tuning/strategy_group.hpp>

using namespace ::testing;
using namespace umpalumpa;
using namespace umpalumpa::data;
using namespace umpalumpa::tuning;

// PREPARATION OF TEST CLASSES

template<typename T = Payload<LogicalDescriptor>> struct DataWrapper : public PayloadWrapper<T>
{
  DataWrapper(std::tuple<T> &t) : PayloadWrapper<T>(t) {}
  DataWrapper(T &d) : PayloadWrapper<T>(d) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
};

struct Settings
{
  void Serialize(std::ostream &) const {}
  static auto Deserialize(std::istream &) { return Settings{}; }
  // Empty
};

class TestOperation_CUDA
  : public BasicOperation<DataWrapper<>, DataWrapper<>, Settings>
  , public KTT_Base
{
public:
  TestOperation_CUDA() : KTT_Base(0) {}

  using KTTStrategy = KTTStrategyBase<OutputData, InputData, Settings>;

  void Synchronize() override {}

  struct MockStrategy : public KTTStrategy
  {
    using KTTStrategy::KTTStrategy;
    std::string GetName() const override { return "MockStrategy"; }
    std::vector<ktt::KernelConfiguration> GetDefaultConfigurations() const override
    {
      return { {} };
    }
    std::unique_ptr<Leader> CreateLeader() const override
    {
      auto uPtr = StrategyGroup::CreateLeader(*this, op);
      auto *ptr = dynamic_cast<MockStrategy *>(uPtr.get());// this can't fail
      EXPECT_CALL(*ptr, GetHash()).WillRepeatedly(Return(mockHash));
      EXPECT_CALL(*ptr, IsSimilarTo).WillRepeatedly(Return(mockSimilar));
      return uPtr;
    }
    tuning::StrategyGroup LoadTuningData() const override
    {
      return tuning::StrategyGroup::LoadTuningData(*this, op);
    }
    // Needed because of creation of a Leader strategy from the MockStrategy
    size_t mockHash = 0;
    bool mockSimilar = false;

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

// NOTE some tests are using knowledge of OperationManager's internal structure, would be good to
// somehow fix it so that the tests are not implementation dependent

class OperationManagerTests : public Test
{
protected:
  // Constructor is called before every test
  OperationManagerTests()
    : settings(), size(42, 1, 1, 1), ld(size),
      pIn(Payload(ld,
        PhysicalDescriptor(nullptr, 0, DataType::Get<float>(), ManagedBy::Manually, nullptr),
        "Input data")),
      pOut(Payload(ld,
        PhysicalDescriptor(nullptr, 0, DataType::Get<float>(), ManagedBy::Manually, nullptr),
        "Output data")),
      in(pIn), out(pOut)
  {
    // NOTE OperationManager is a singleton and therefore has a global state. It needs to be reset
    // before each test.
    StrategyManager::Get().Cleanup();
  }

  const Settings settings;
  const Size size;
  const LogicalDescriptor ld;

  Payload<LogicalDescriptor> pIn;
  Payload<LogicalDescriptor> pOut;

  const TestOperation_CUDA::InputData in;
  const TestOperation_CUDA::OutputData out;
};

TEST_F(OperationManagerTests, strategy_registered_automatically_when_init_succeeds)
{
  TestOperation_CUDA opo;
  EXPECT_CALL(*opo.mockStratPtr, InitImpl()).WillOnce(Return(true));

  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(opo.Init(out, in, settings));
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 1u);
}

TEST_F(OperationManagerTests, strategy_not_registered_when_init_failed)
{
  TestOperation_CUDA opo;
  EXPECT_CALL(*opo.mockStratPtr, InitImpl()).WillOnce(Return(false));

  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
  ASSERT_FALSE(opo.Init(out, in, settings));
  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
}

TEST_F(OperationManagerTests, strategy_unregistered_automatically_when_destroyed)
{
  {
    TestOperation_CUDA opo;
    EXPECT_CALL(*opo.mockStratPtr, InitImpl()).WillOnce(Return(true));

    ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
    ASSERT_TRUE(opo.Init(out, in, settings));
    ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
    ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 1u);
  }// the operation and the strategy are destroyed

  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.empty());
}

TEST_F(OperationManagerTests, strategy_already_registered_cant_register_again)
{
  TestOperation_CUDA opo;
  EXPECT_CALL(*opo.mockStratPtr, InitImpl()).WillOnce(Return(true));

  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(opo.Init(out, in, settings));
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
  // Now we have 1 strategy registered and now we try to insert again the same strategy
  // To force this situation we use explicit call to Register method
  StrategyManager::Get().Register(*opo.mockStratPtr);// Try to add the same strategy again
  // Nothing should change
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 1u);
}

TEST_F(OperationManagerTests, not_registered_strategy_cant_be_unregistered)
{
  TestOperation_CUDA opo;

  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
  StrategyManager::Get().Unregister(*opo.mockStratPtr);// Should print a warning
  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
}

TEST_F(OperationManagerTests,
  GetBestConfiguration_returns_empty_configuration_when_passed_invalid_hash)
{
  // OperationManager is empty therefore every hash is invalid one
  // When DB is added, it needs to be mocked in tests
  // ASSERT_TRUE(OperationManager::Get().GetBestConfiguration(0).GetPairs().empty());
}

TEST_F(OperationManagerTests,
  strategy_registered_GetBestConfiguration_returns_best_config_when_passed_valid_hash)
{
  TestOperation_CUDA opo;
  EXPECT_CALL(*opo.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*opo.mockStratPtr, Execute).WillOnce(Return(true));
  EXPECT_CALL(*opo.mockStratPtr, GetHash()).WillRepeatedly(Return(42));
  opo.mockStratPtr->mockHash = 42;

  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(opo.Init(out, in, settings));
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 1u);
  // We need to execute the operation, so that KTT can generate some config
  // TODO either properly define TestOperation_CUDA to use KTT or generate the config in some other
  // way
  ASSERT_TRUE(opo.Execute(out, in));

  // auto bestConfig = OperationManager::Get().GetBestConfiguration(opo.mockStratPtr->GetHash());
  // TODO some assert
}

TEST_F(OperationManagerTests,
  strategy_not_registered_GetBestConfiguration_returns_best_config_when_passed_valid_hash)
{
  // TODO when DB is added (DB should be mocked here!!!)
}

TEST_F(OperationManagerTests, multiple_equivalent_strategies_registered_at_once)
{
  TestOperation_CUDA opo1;
  TestOperation_CUDA opo2;
  TestOperation_CUDA opo3;
  EXPECT_CALL(*opo1.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*opo2.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*opo3.mockStratPtr, InitImpl()).WillOnce(Return(true));

  // Same hash == equivalence
  EXPECT_CALL(*opo1.mockStratPtr, GetHash()).WillRepeatedly(Return(42));
  EXPECT_CALL(*opo2.mockStratPtr, GetHash()).WillRepeatedly(Return(42));
  EXPECT_CALL(*opo3.mockStratPtr, GetHash()).WillRepeatedly(Return(42));
  opo1.mockStratPtr->mockHash = 42;
  opo2.mockStratPtr->mockHash = 42;
  opo3.mockStratPtr->mockHash = 42;

  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(opo1.Init(out, in, settings));
  ASSERT_TRUE(opo2.Init(out, in, settings));
  ASSERT_TRUE(opo3.Init(out, in, settings));
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 3u);
}

TEST_F(OperationManagerTests, multiple_similar_strategies_registered_at_once)
{
  TestOperation_CUDA opo1;
  TestOperation_CUDA opo2;
  TestOperation_CUDA opo3;
  EXPECT_CALL(*opo1.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*opo2.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*opo3.mockStratPtr, InitImpl()).WillOnce(Return(true));

  // Different hash != equivalence
  EXPECT_CALL(*opo1.mockStratPtr, GetHash()).WillRepeatedly(Return(123));
  EXPECT_CALL(*opo2.mockStratPtr, GetHash()).WillRepeatedly(Return(456));
  EXPECT_CALL(*opo3.mockStratPtr, GetHash()).WillRepeatedly(Return(789));
  opo1.mockStratPtr->mockHash = 123;
  opo2.mockStratPtr->mockHash = 456;
  opo3.mockStratPtr->mockHash = 789;

  // But they are similar
  EXPECT_CALL(*opo1.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));
  EXPECT_CALL(*opo2.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));
  EXPECT_CALL(*opo3.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));
  opo1.mockStratPtr->mockSimilar = true;
  opo2.mockStratPtr->mockSimilar = true;
  opo3.mockStratPtr->mockSimilar = true;

  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(opo1.Init(out, in, settings));
  ASSERT_TRUE(opo2.Init(out, in, settings));
  ASSERT_TRUE(opo3.Init(out, in, settings));
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 3u);
}

TEST_F(OperationManagerTests, multiple_different_strategies_registered_at_once)
{
  TestOperation_CUDA opo1;
  TestOperation_CUDA opo2;
  TestOperation_CUDA opo3;
  EXPECT_CALL(*opo1.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*opo2.mockStratPtr, InitImpl()).WillOnce(Return(true));
  EXPECT_CALL(*opo3.mockStratPtr, InitImpl()).WillOnce(Return(true));

  // Different hash != equivalence
  EXPECT_CALL(*opo1.mockStratPtr, GetHash()).WillRepeatedly(Return(123));
  EXPECT_CALL(*opo2.mockStratPtr, GetHash()).WillRepeatedly(Return(456));
  EXPECT_CALL(*opo3.mockStratPtr, GetHash()).WillRepeatedly(Return(789));
  opo1.mockStratPtr->mockHash = 123;
  opo2.mockStratPtr->mockHash = 456;
  opo3.mockStratPtr->mockHash = 789;

  // They are not similar
  EXPECT_CALL(*opo1.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));
  EXPECT_CALL(*opo2.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));
  EXPECT_CALL(*opo3.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));
  opo1.mockStratPtr->mockSimilar = false;
  opo2.mockStratPtr->mockSimilar = false;
  opo3.mockStratPtr->mockSimilar = false;

  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
  ASSERT_TRUE(opo1.Init(out, in, settings));
  ASSERT_TRUE(opo2.Init(out, in, settings));
  ASSERT_TRUE(opo3.Init(out, in, settings));
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 3u);
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 1u);
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(1).strategies.size(), 1u);
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(2).strategies.size(), 1u);
}

TEST_F(OperationManagerTests, multiple_similar_strategies_unregistered_correctly)
{
  {
    TestOperation_CUDA opo1;
    {
      TestOperation_CUDA opo2;
      {
        TestOperation_CUDA opo3;

        EXPECT_CALL(*opo1.mockStratPtr, InitImpl()).WillOnce(Return(true));
        EXPECT_CALL(*opo2.mockStratPtr, InitImpl()).WillOnce(Return(true));
        EXPECT_CALL(*opo3.mockStratPtr, InitImpl()).WillOnce(Return(true));

        // Different hash != equivalence
        EXPECT_CALL(*opo1.mockStratPtr, GetHash()).WillRepeatedly(Return(123));
        EXPECT_CALL(*opo2.mockStratPtr, GetHash()).WillRepeatedly(Return(456));
        EXPECT_CALL(*opo3.mockStratPtr, GetHash()).WillRepeatedly(Return(789));
        opo1.mockStratPtr->mockHash = 123;
        opo2.mockStratPtr->mockHash = 456;
        opo3.mockStratPtr->mockHash = 789;

        // But they are similar
        EXPECT_CALL(*opo1.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));
        EXPECT_CALL(*opo2.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));
        EXPECT_CALL(*opo3.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(true));
        opo1.mockStratPtr->mockSimilar = true;
        opo2.mockStratPtr->mockSimilar = true;
        opo3.mockStratPtr->mockSimilar = true;

        ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
        ASSERT_TRUE(opo1.Init(out, in, settings));
        ASSERT_TRUE(opo2.Init(out, in, settings));
        ASSERT_TRUE(opo3.Init(out, in, settings));
        ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
        ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 3u);
      }// opo3 is destroyed, therefore its strategy is unregistered
      ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
      ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 2u);
    }// opo2 is destroyed, therefore its strategy is unregistered
    ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
    ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 1u);
  }// opo1 is destroyed, therefore its strategy is unregistered
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 1u);
  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.empty());
}

TEST_F(OperationManagerTests, multiple_different_strategies_unregistered_correctly)
{
  {
    TestOperation_CUDA opo1;
    {
      TestOperation_CUDA opo2;
      {
        TestOperation_CUDA opo3;

        EXPECT_CALL(*opo1.mockStratPtr, InitImpl()).WillOnce(Return(true));
        EXPECT_CALL(*opo2.mockStratPtr, InitImpl()).WillOnce(Return(true));
        EXPECT_CALL(*opo3.mockStratPtr, InitImpl()).WillOnce(Return(true));

        // Different hash != equivalence
        EXPECT_CALL(*opo1.mockStratPtr, GetHash()).WillRepeatedly(Return(123));
        EXPECT_CALL(*opo2.mockStratPtr, GetHash()).WillRepeatedly(Return(456));
        EXPECT_CALL(*opo3.mockStratPtr, GetHash()).WillRepeatedly(Return(789));
        opo1.mockStratPtr->mockHash = 123;
        opo2.mockStratPtr->mockHash = 456;
        opo3.mockStratPtr->mockHash = 789;

        // They are not similar
        EXPECT_CALL(*opo1.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));
        EXPECT_CALL(*opo2.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));
        EXPECT_CALL(*opo3.mockStratPtr, IsSimilarTo).WillRepeatedly(Return(false));
        opo1.mockStratPtr->mockSimilar = false;
        opo2.mockStratPtr->mockSimilar = false;
        opo3.mockStratPtr->mockSimilar = false;

        ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().empty());
        ASSERT_TRUE(opo1.Init(out, in, settings));
        ASSERT_TRUE(opo2.Init(out, in, settings));
        ASSERT_TRUE(opo3.Init(out, in, settings));
        ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 3u);
        ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 1u);
        ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(1).strategies.size(), 1u);
        ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(2).strategies.size(), 1u);
      }// opo3 is destroyed, therefore its strategy is unregistered
      ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 3u);
      ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 1u);
      ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(1).strategies.size(), 1u);
      ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().at(2).strategies.empty());
    }// opo2 is destroyed, therefore its strategy is unregistered
    ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 3u);
    ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.size(), 1u);
    ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().at(1).strategies.empty());
    ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().at(2).strategies.empty());
  }// opo1 is destroyed, therefore its strategy is unregistered
  ASSERT_EQ(StrategyManager::Get().GetRegisteredStrategies().size(), 3u);
  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().at(0).strategies.empty());
  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().at(1).strategies.empty());
  ASSERT_TRUE(StrategyManager::Get().GetRegisteredStrategies().at(2).strategies.empty());
}
