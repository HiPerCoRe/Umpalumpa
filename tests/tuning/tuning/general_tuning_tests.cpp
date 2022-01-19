#include <gtest/gtest.h>
#include <tests/tuning/waiting_algorithm.hpp>

// tests:
// strategy never tuned -> run default config
// strategy is tuning -> run various configs
// strategy already tuned -> run best found config
// strategy tuned -> run best -> continue tuning -> run best
// multiple strategies...

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
};

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
