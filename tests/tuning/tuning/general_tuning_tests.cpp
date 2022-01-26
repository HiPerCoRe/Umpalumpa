#include <gtest/gtest.h>
#include <tests/tuning/waiting_algorithm.hpp>
#include <iostream>

using namespace umpalumpa;
using namespace umpalumpa::data;
using namespace umpalumpa::tuning;

class GeneralTuningTests : public ::testing::Test
{
public:
  GeneralTuningTests()
    : pData(LogicalDescriptor(Size(1, 1, 1, 1)),
      PhysicalDescriptor(nullptr, 0, DataType::kVoid, ManagedBy::Manually, nullptr)),
      in(pData), out(pData)
  {}

  // void TearDown() override { StrategyManager::Get().Cleanup(); }

protected:
  Settings settings{ 0, 0 };// { equalityGroup, similarityGroup }
  Payload<LogicalDescriptor> pData;
  WaitingAlgorithm::InputData in;
  WaitingAlgorithm::InputData out;

  WaitingAlgorithm alg = WaitingAlgorithm(0);

  constexpr static auto manualCheckMsg = "\n\t## Manual check -- ";
};

TEST_F(GeneralTuningTests, no_tuning_should_run_default_config)
{
  settings = { 1, 1 };

  ASSERT_TRUE(alg.Init(out, in, settings));
  std::cout << manualCheckMsg << "Expecting default config (1 time):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
}

TEST_F(GeneralTuningTests, no_tuning_should_run_default_config_multiple_times)
{
  settings = { 2, 2 };

  ASSERT_TRUE(alg.Init(out, in, settings));
  std::cout << manualCheckMsg << "Expecting default configs (3 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));
}

TEST_F(GeneralTuningTests, tuning_is_on_various_configs_should_be_tried)
{
  settings = { 3, 3 };

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(alg.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);

  std::cout << manualCheckMsg << "Expecting various configs (3 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));
}

TEST_F(GeneralTuningTests, tune_strategy_then_run_best_found_config)
{
  settings = { 4, 4 };

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(alg.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);

  // Tuning executes
  std::cout << manualCheckMsg << "Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));

  // Turn off tuning
  strat.SetTuningApproach(TuningApproach::kNoTuning);

  // Best config executes
  std::cout << manualCheckMsg << "Expecting the best found configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));
}

TEST_F(GeneralTuningTests, tune_then_run_best_tune_again_then_run_best)
{
  settings = { 5, 5 };

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(alg.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);

  // First tuning executes
  std::cout << manualCheckMsg << "Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));

  // Turn off tuning
  strat.SetTuningApproach(TuningApproach::kNoTuning);

  // Best config (of first tuning) executes
  std::cout << manualCheckMsg << "Expecting the best found configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));

  // Turn on tuning
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);

  // Second tuning executes
  std::cout << manualCheckMsg << "Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));

  // Turn off tuning
  strat.SetTuningApproach(TuningApproach::kNoTuning);

  // Best config (of all tunings) executes
  std::cout << manualCheckMsg << "Expecting the best found configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));
}

TEST_F(GeneralTuningTests, two_equal_strategies_both_should_tune)
{
  // Set Settings
  // We want equal strategies, therefore we use the same settings
  settings = Settings{ 100, 100 };

  WaitingAlgorithm alg2 = WaitingAlgorithm(0);

  ASSERT_TRUE(alg.Init(out, in, settings));
  ASSERT_TRUE(alg2.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(alg.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);
  auto &strat2 = dynamic_cast<TunableStrategy &>(alg2.GetStrategy());
  strat2.SetTuningApproach(TuningApproach::kEntireStrategy);

  // Both algorithms should be tuning the same stuff
  // (tuned configs might overlap because the algorithms know nothing about each other)
  std::cout << manualCheckMsg << "Alg1 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));
  std::cout << manualCheckMsg << "Alg2 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(alg2.Execute(out, in));
  ASSERT_TRUE(alg2.Execute(out, in));

  // Turn off tuning
  strat.SetTuningApproach(TuningApproach::kNoTuning);
  strat2.SetTuningApproach(TuningApproach::kNoTuning);

  // Both algorithms should have the same best config
  std::cout << manualCheckMsg << "Alg1 Expecting the best found configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));
  std::cout << manualCheckMsg << "Alg2 Expecting the best found configs (2 times):" << std::endl;
  ASSERT_TRUE(alg2.Execute(out, in));
  ASSERT_TRUE(alg2.Execute(out, in));
}

TEST_F(GeneralTuningTests, two_similar_strategies_only_leader_can_tune)
{
  // Set Settings
  // We want similar strategies, therefore we change equalityGroup
  settings = Settings{ 101, 101 };// { equalityGroup, similarityGroup }
  auto settings2 = Settings{ 102, 101 };// { equalityGroup, similarityGroup }

  WaitingAlgorithm alg2 = WaitingAlgorithm(0);

  ASSERT_TRUE(alg.Init(out, in, settings));
  ASSERT_TRUE(alg2.Init(out, in, settings2));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(alg.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);
  auto &strat2 = dynamic_cast<TunableStrategy &>(alg2.GetStrategy());
  strat2.SetTuningApproach(TuningApproach::kEntireStrategy);

  // This should correctly run tuning of the kernels
  std::cout << manualCheckMsg << "Alg1 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));

  // Should run the best config even though it is set to run a tuning
  // because it is not a strategyGroup Leader and it is not equal to the Leader (it is just similar)
  // This has been set in the Settings
  std::cout << manualCheckMsg
            << "Alg2 Expecting best configs, even though tuning is enabled (2 times):" << std::endl;
  ASSERT_TRUE(alg2.Execute(out, in));
  ASSERT_TRUE(alg2.Execute(out, in));
}

TEST_F(GeneralTuningTests, two_similar_strategies_reuse_tuned_config)
{
  // Set Settings
  // We want similar strategies, therefore we change equalityGroup
  settings = Settings{ 105, 105 };// { equalityGroup, similarityGroup }
  auto settings2 = Settings{ 106, 105 };// { equalityGroup, similarityGroup }

  WaitingAlgorithm alg2 = WaitingAlgorithm(0);

  ASSERT_TRUE(alg.Init(out, in, settings));
  ASSERT_TRUE(alg2.Init(out, in, settings2));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(alg.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);
  auto &strat2 = dynamic_cast<TunableStrategy &>(alg2.GetStrategy());
  strat2.SetTuningApproach(TuningApproach::kNoTuning);

  // Tune the strategy
  std::cout << manualCheckMsg << "Alg1 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));

  // Similar strategy should use the tuned config of the leader
  std::cout << manualCheckMsg << "Alg2 Expecting the best found config (1 time):" << std::endl;
  ASSERT_TRUE(alg2.Execute(out, in));

  // Tune the strategy
  std::cout << manualCheckMsg << "Alg1 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));

  // Similar strategy should use the tuned config of the leader
  std::cout << manualCheckMsg << "Alg2 Expecting the best found config (1 time):" << std::endl;
  ASSERT_TRUE(alg2.Execute(out, in));
}

TEST_F(GeneralTuningTests, same_strategies_different_settings_do_not_interfere)
{
  // Set Settings
  // We want different strategies, therefore we use different settings
  settings = Settings{ 110, 110 };
  auto settings2 = Settings{ 111, 111 };

  WaitingAlgorithm alg2 = WaitingAlgorithm(0);

  ASSERT_TRUE(alg.Init(out, in, settings));
  ASSERT_TRUE(alg2.Init(out, in, settings2));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(alg.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);
  auto &strat2 = dynamic_cast<TunableStrategy &>(alg2.GetStrategy());
  strat2.SetTuningApproach(TuningApproach::kEntireStrategy);

  // Each algorithm is tuning its own strategy
  std::cout << manualCheckMsg << "Alg1 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  ASSERT_TRUE(alg.Execute(out, in));
  std::cout << manualCheckMsg << "Alg2 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(alg2.Execute(out, in));
  ASSERT_TRUE(alg2.Execute(out, in));

  // Turn off tuning
  strat.SetTuningApproach(TuningApproach::kNoTuning);
  strat2.SetTuningApproach(TuningApproach::kNoTuning);

  // Each algorithm should have different best config (unless they got tuned to the same value)
  std::cout << manualCheckMsg
            << "Alg1 Expecting the best found config of the first tuning:" << std::endl;
  ASSERT_TRUE(alg.Execute(out, in));
  std::cout << manualCheckMsg
            << "Alg2 Expecting the best found config of the second tuning:" << std::endl;
  ASSERT_TRUE(alg2.Execute(out, in));
}

// tests:
// multiple different strategies (should not interfere, can tune just specified
// subset)

/*
TEST_F(GeneralTuningTests, TODOname)
{
  // Set Settings
  // settings.equalityGroup = ...;

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  // auto &strat = dynamic_cast<TunableStrategy &>(alg.GetStrategy());
  // strat.SetTuningApproach(TuningApproach::kEntireStrategy);

  ASSERT_TRUE(alg.Execute(out, in));

  // TODO checks
}
*/

