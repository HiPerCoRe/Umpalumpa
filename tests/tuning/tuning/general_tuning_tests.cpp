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

// tests:
// multiple same strategies (should not interfere if not equal, should reuse tuned stuff when
// equal/similar)
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

