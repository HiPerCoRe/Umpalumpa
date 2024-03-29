#include <gtest/gtest.h>
#include <tests/tuning/waiting_operation.hpp>
#include <iostream>

using namespace umpalumpa;
using namespace umpalumpa::data;
using namespace umpalumpa::tuning;

class GeneralTuningTests : public ::testing::Test
{
public:
  GeneralTuningTests()
    : pData(LogicalDescriptor(Size(1, 1, 1, 1)),
      PhysicalDescriptor(nullptr, 0, DataType::Get<void>(), ManagedBy::Manually, nullptr)),
      in(pData), out(pData)
  {}

protected:
  Settings settings{ 0, 0, 1 };// { equalityGroup, similarityGroup }
  Payload<LogicalDescriptor> pData;
  WaitingOperation::InputData in;
  WaitingOperation::OutputData out;

  WaitingOperation op = WaitingOperation(0);

  constexpr static auto manualCheckMsg = "\n\t## Manual check -- ";
};

TEST_F(GeneralTuningTests, OneStrategy_NotTuned_RunsDefault)
{
  settings = { 1, 1 };

  ASSERT_TRUE(op.Init(out, in, settings));
  std::cout << manualCheckMsg << "Expecting default config (1 time):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
}

TEST_F(GeneralTuningTests, OneStrategy_NotTuned_RunsDefault_3Times)
{
  settings = { 2, 2 };

  ASSERT_TRUE(op.Init(out, in, settings));
  std::cout << manualCheckMsg << "Expecting default configs (3 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
}

TEST_F(GeneralTuningTests, OneStrategy_Tuning)
{
  settings = { 3, 3 };

  ASSERT_TRUE(op.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(op.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);

  std::cout << manualCheckMsg << "Expecting various configs (3 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
}

TEST_F(GeneralTuningTests, OneStrategy_Tune_ThenRunBestFound)
{
  settings = { 4, 4 };

  ASSERT_TRUE(op.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(op.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);

  // Tuning executes
  std::cout << manualCheckMsg << "Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));

  // Turn off tuning
  strat.SetTuningApproach(TuningApproach::kNoTuning);

  // Best config executes
  std::cout << manualCheckMsg << "Expecting the best found configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
}

TEST_F(GeneralTuningTests, OneStrategy_Tune_ThenRunBestFound_ThenRetune_ThenRunBestFound)
{
  settings = { 5, 5 };

  ASSERT_TRUE(op.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(op.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);

  // First tuning executes
  std::cout << manualCheckMsg << "Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));

  // Turn off tuning
  strat.SetTuningApproach(TuningApproach::kNoTuning);

  // Best config (of first tuning) executes
  std::cout << manualCheckMsg << "Expecting the best found configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));

  // Turn on tuning
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);

  // Second tuning executes
  std::cout << manualCheckMsg << "Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));

  // Turn off tuning
  strat.SetTuningApproach(TuningApproach::kNoTuning);

  // Best config (of all tunings) executes
  std::cout << manualCheckMsg << "Expecting the best found configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
}

TEST_F(GeneralTuningTests, TwoEqualStrategies_BothTune)
{
  // Set Settings
  // We want equal strategies, therefore we use the same settings
  settings = Settings{ 100, 100 };

  WaitingOperation op2 = WaitingOperation(0);

  ASSERT_TRUE(op.Init(out, in, settings));
  ASSERT_TRUE(op2.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(op.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);
  auto &strat2 = dynamic_cast<TunableStrategy &>(op2.GetStrategy());
  strat2.SetTuningApproach(TuningApproach::kEntireStrategy);

  // Both operations should be tuning the same stuff
  // (tuned configs might overlap because the operations know nothing about each other)
  std::cout << manualCheckMsg << "Op1 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
  std::cout << manualCheckMsg << "Op2 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(op2.Execute(out, in));
  ASSERT_TRUE(op2.Execute(out, in));

  // Turn off tuning
  strat.SetTuningApproach(TuningApproach::kNoTuning);
  strat2.SetTuningApproach(TuningApproach::kNoTuning);

  // Both operations should have the same best config
  std::cout << manualCheckMsg << "Op1 Expecting the best found configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
  std::cout << manualCheckMsg << "Op2 Expecting the best found configs (2 times):" << std::endl;
  ASSERT_TRUE(op2.Execute(out, in));
  ASSERT_TRUE(op2.Execute(out, in));
}

TEST_F(GeneralTuningTests, TwoSimilarStrategies_OnlyLeaderCanTune)
{
  // Set Settings
  // We want similar strategies, therefore we change equalityGroup
  settings = Settings{ 101, 101 };// { equalityGroup, similarityGroup }
  auto settings2 = Settings{ 102, 101 };// { equalityGroup, similarityGroup }

  WaitingOperation op2 = WaitingOperation(0);

  ASSERT_TRUE(op.Init(out, in, settings));
  ASSERT_TRUE(op2.Init(out, in, settings2));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(op.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);
  auto &strat2 = dynamic_cast<TunableStrategy &>(op2.GetStrategy());
  strat2.SetTuningApproach(TuningApproach::kEntireStrategy);

  // This should correctly run tuning of the kernels
  std::cout << manualCheckMsg << "Op1 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));

  // Should run the best config even though it is set to run a tuning
  // because it is not a strategyGroup Leader and it is not equal to the Leader (it is just similar)
  // This has been set in the Settings
  std::cout << manualCheckMsg
            << "Op2 Expecting best configs, even though tuning is enabled (2 times):" << std::endl;
  ASSERT_TRUE(op2.Execute(out, in));
  ASSERT_TRUE(op2.Execute(out, in));
}

TEST_F(GeneralTuningTests, TwoSimilarStrategies_UseLeadersConfig)
{
  // Set Settings
  // We want similar strategies, therefore we change equalityGroup
  settings = Settings{ 105, 105 };// { equalityGroup, similarityGroup }
  auto settings2 = Settings{ 106, 105 };// { equalityGroup, similarityGroup }

  WaitingOperation op2 = WaitingOperation(0);

  ASSERT_TRUE(op.Init(out, in, settings));
  ASSERT_TRUE(op2.Init(out, in, settings2));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(op.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);
  auto &strat2 = dynamic_cast<TunableStrategy &>(op2.GetStrategy());
  strat2.SetTuningApproach(TuningApproach::kNoTuning);

  // Tune the strategy
  std::cout << manualCheckMsg << "Op1 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));

  // Similar strategy should use the tuned config of the leader
  std::cout << manualCheckMsg << "Op2 Expecting the best found config (1 time):" << std::endl;
  ASSERT_TRUE(op2.Execute(out, in));

  // Tune the strategy
  std::cout << manualCheckMsg << "Op1 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));

  // Similar strategy should use the tuned config of the leader
  std::cout << manualCheckMsg << "Op2 Expecting the best found config (1 time):" << std::endl;
  ASSERT_TRUE(op2.Execute(out, in));
}

TEST_F(GeneralTuningTests, TwoStrategies_TuningNotInterfering)
{
  // Set Settings
  // We want different strategies, therefore we use different settings
  settings = Settings{ 110, 110 };
  auto settings2 = Settings{ 111, 111 };

  WaitingOperation op2 = WaitingOperation(0);

  ASSERT_TRUE(op.Init(out, in, settings));
  ASSERT_TRUE(op2.Init(out, in, settings2));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(op.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);
  auto &strat2 = dynamic_cast<TunableStrategy &>(op2.GetStrategy());
  strat2.SetTuningApproach(TuningApproach::kEntireStrategy);

  // Each operation is tuning its own strategy
  std::cout << manualCheckMsg << "Op1 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
  std::cout << manualCheckMsg << "Op2 Expecting various configs (2 times):" << std::endl;
  ASSERT_TRUE(op2.Execute(out, in));
  ASSERT_TRUE(op2.Execute(out, in));

  // Turn off tuning
  strat.SetTuningApproach(TuningApproach::kNoTuning);
  strat2.SetTuningApproach(TuningApproach::kNoTuning);

  // Each operation should have different best config (unless they got tuned to the same value)
  std::cout << manualCheckMsg
            << "Op1 Expecting the best found config of the first tuning:" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  std::cout << manualCheckMsg
            << "Op2 Expecting the best found config of the second tuning:" << std::endl;
  ASSERT_TRUE(op2.Execute(out, in));
}

TEST_F(GeneralTuningTests, OneStrategy_MultipleKernels_AllTuning)
{
  // Set Settings
  settings = Settings{ 200, 200, 3 };// { equalityGroup, similarityGroup, numberOfKernels }

  ASSERT_TRUE(op.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  // Turn on tuning for all kernels
  auto &strat = dynamic_cast<TunableStrategy &>(op.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);

  std::cout << manualCheckMsg << "Expecting tuning of " << settings.numberOfKernels
            << " kernels (2 times):" << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
}

TEST_F(GeneralTuningTests, OneStrategy_MultipleKernels_SelectedTuning)
{
  // Set Settings
  settings = Settings{ 201, 201, 3 };

  ASSERT_TRUE(op.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  // Turn on tuning for kernels: 0, 2
  auto &strat = dynamic_cast<TunableStrategy &>(op.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kSelectedKernels);
  strat.SetTuningForIdx(0, true);
  strat.SetTuningForIdx(2, true);

  std::cout << manualCheckMsg
            << "Expecting tuning of kernels 0,2 and default config of kernel 1 (2 times):"
            << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
}

TEST_F(GeneralTuningTests, OneStrategy_MultipleKernels_NotInterferingWithEachOther)
{
  // Set Settings
  settings = Settings{ 202, 202, 3 };

  ASSERT_TRUE(op.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  // Turn on tuning for kernels: 0, 2
  auto &strat = dynamic_cast<TunableStrategy &>(op.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kSelectedKernels);
  strat.SetTuningForIdx(0, true);
  strat.SetTuningForIdx(2, true);

  std::cout << manualCheckMsg
            << "Expecting tuning of kernels 0,2 and default config of kernel 1 (2 times):"
            << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));

  // Turn tuning off
  strat.SetTuningApproach(TuningApproach::kNoTuning);

  std::cout << manualCheckMsg
            << "Expecting best configs of kernels 0,2 and default config of kernel 1 (2 times):"
            << std::endl;
  ASSERT_TRUE(op.Execute(out, in));
  ASSERT_TRUE(op.Execute(out, in));
}

TEST_F(GeneralTuningTests, Serialization)
{
  StrategyManager::Get().Cleanup();

  settings = { 123, 123, 1 };

  ASSERT_TRUE(op.Init(out, in, settings));
  // After successful Init there is a strategy to work with!

  auto &strat = dynamic_cast<TunableStrategy &>(op.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kEntireStrategy);

  ASSERT_TRUE(op.Execute(out, in));
  strat.SetTuningApproach(TuningApproach::kNoTuning);
  ASSERT_TRUE(op.Execute(out, in));

  std::cout << "Saving\n";
  StrategyManager::Get().SaveTuningData();

  std::cout << "Loading\n";
  auto x = strat.LoadTuningData();

  std::cout << "Printing\n";
  x.leader->Serialize(std::cout);
}

