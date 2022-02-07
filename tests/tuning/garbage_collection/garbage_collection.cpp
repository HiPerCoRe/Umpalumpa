#include <gtest/gtest.h>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/tuning/ktt_helper.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>
#include <thread>

using namespace umpalumpa;
using namespace umpalumpa::utils;
using namespace umpalumpa::tuning;
using KTTIdTracker = umpalumpa::tuning::KTTIdTracker;

using namespace ::testing;

// clang-format off
#define ASSERT_KTT_ERROR_MSG(cmd) \
    internal::CaptureStderr();\
    cmd;\
    {auto output = internal::GetCapturedStderr(); \
    ASSERT_TRUE(StartsWith(output, "[ERROR]")) << "Output is: '" << output << "'";}
// clang-format on

class TestStrategy : public TunableStrategy
{
  const std::string kernelFile;

public:
  TestStrategy(KTTHelper &helper, const std::string &file)
    : TunableStrategy(helper), kernelFile(file)
  {}

  ~TestStrategy()
  {
    spdlog::debug("Destructor: dId-{0} kId-{1} aId-{2}",
      testDefinitionIds.back(),
      testKernelIds.back(),
      testArgumentIds.back());
  }

  bool Init(const std::string &kernelName)
  {
    std::lock_guard lck(kttHelper.GetMutex());
    Cleanup();// cleanup before initializing

    AddKernelDefinition(kernelName, kernelFile, {});
    AddKernel(kernelName, GetDefinitionId());
    testDefinitionIds.push_back(GetDefinitionId());
    testKernelIds.push_back(GetKernelId());
    spdlog::debug("Init: dId-{0} kId-{1}", testDefinitionIds.back(), testKernelIds.back());
    return true;
  }

  bool Execute()
  {
    std::lock_guard lck(kttHelper.GetMutex());
    auto &tuner = kttHelper.GetTuner();
    auto argumentId = tuner.AddArgumentScalar(NULL);
    testArgumentIds.push_back(argumentId);
    SetArguments(GetDefinitionId(), { argumentId });
    spdlog::debug("Execute: dId-{0} kId-{1} aId-{2}",
      testDefinitionIds.back(),
      testKernelIds.back(),
      testArgumentIds.back());
    return tuner.Run(GetKernelId(), {}, {}).IsValid();
  }

  size_t GetHash() const override { return 0; }
  std::vector<ktt::KernelConfiguration> GetDefaultConfigurations() const override { return { {} }; }
  std::unique_ptr<Leader> CreateLeader() const override { return std::unique_ptr<Leader>(nullptr); }
  std::vector<tuning::StrategyGroup> Tmp() const override { return {}; }
  bool IsSimilarTo(const TunableStrategy &) const override { return false; }

  std::vector<ktt::KernelDefinitionId> testDefinitionIds;
  std::vector<ktt::KernelId> testKernelIds;
  std::vector<ktt::ArgumentId> testArgumentIds;
};

class GarbageCollectionTests : public Test
{
protected:
  GarbageCollectionTests()
    : baseAlgo(0), kttHelper(baseAlgo.GetHelper()), tuner(kttHelper.GetTuner())
  {}

  KTT_Base baseAlgo;
  KTTHelper &kttHelper;
  ktt::Tuner &tuner;

  static const inline std::string kernelFile =
    utils::GetSourceFilePath("tests/tuning/garbage_collection/test_kernels.cu");
  static const inline std::string kernelName1 = "TestKernel1";
  static const inline std::string kernelName2 = "TestKernel2";
  static const inline std::string kernelName3 = "TestKernel3";

  bool StartsWith(const std::string &s, const std::string &prefix) const
  {
    return s.rfind(prefix, 0) == 0;
  }
};

// need to check that the strat's ids are in KTT -> execute should be enough for this
// need to check that the ids are removed after LAST strategy with the same Id tracker dies

TEST_F(GarbageCollectionTests, when_strategy_destroyed_ids_are_released)
{
  ktt::KernelDefinitionId definitionId;
  ktt::KernelId kernelId;
  ktt::ArgumentId argumentId;
  {
    TestStrategy strat(kttHelper, kernelFile);
    // Strategy uses 1 definition id and 1 kernel id
    ASSERT_TRUE(strat.Init(kernelName1));
    // Strategy uses 1 argument id
    ASSERT_TRUE(strat.Execute());
    // Execute returning true and passing the assert means that all the used ids are valid
    // We can additionally check it
    definitionId = strat.testDefinitionIds.back();
    kernelId = strat.testKernelIds.back();
    argumentId = strat.testArgumentIds.back();
    ASSERT_NE(definitionId, ktt::InvalidKernelDefinitionId);
    ASSERT_NE(kernelId, ktt::InvalidKernelId);
    ASSERT_NE(argumentId, ktt::InvalidArgumentId);
  }// TestStrategy is destroyed, all the ids should be removed

  // Definition id is removed
  ASSERT_KTT_ERROR_MSG(tuner.CreateSimpleKernel(kernelName2, definitionId));
  // Kernel id is removed
  ASSERT_KTT_ERROR_MSG(tuner.Run(kernelId, {}, {}));
  // Argument id is removed
  auto validDefinitionId = tuner.AddKernelDefinitionFromFile(kernelName3, kernelFile, {}, {});
  ASSERT_NE(validDefinitionId, ktt::InvalidKernelDefinitionId);
  ASSERT_KTT_ERROR_MSG(tuner.SetArguments(validDefinitionId, { argumentId }));
  // Cleanup tuner
  tuner.RemoveKernelDefinition(validDefinitionId);
}

TEST_F(GarbageCollectionTests, strategies_with_same_id_are_released_after_last_is_destroyed)
{
  ktt::KernelDefinitionId definitionId;
  ktt::KernelId kernelId;
  ktt::KernelId kernelId2;
  ktt::ArgumentId argumentId;
  ktt::ArgumentId argumentId2;
  {
    TestStrategy strat(kttHelper, kernelFile);
    {
      TestStrategy strat2(kttHelper, kernelFile);
      // Strategy uses 1 definition id and 1 kernel id
      ASSERT_TRUE(strat.Init(kernelName1));
      // Strategy uses 1 argument id
      ASSERT_TRUE(strat.Execute());
      // Execute returning true and passing the assert means that all the used ids are valid
      // We can additionally check it
      definitionId = strat.testDefinitionIds.back();
      kernelId = strat.testKernelIds.back();
      argumentId = strat.testArgumentIds.back();
      ASSERT_NE(definitionId, ktt::InvalidKernelDefinitionId);
      ASSERT_NE(kernelId, ktt::InvalidKernelId);
      ASSERT_NE(argumentId, ktt::InvalidArgumentId);
      // Initialize second strat
      ASSERT_TRUE(strat2.Init(kernelName1));
      ASSERT_TRUE(strat2.Execute());
      kernelId2 = strat2.testKernelIds.back();
      argumentId2 = strat2.testArgumentIds.back();
      ASSERT_NE(kernelId2, ktt::InvalidKernelId);
      ASSERT_NE(argumentId2, ktt::InvalidArgumentId);
    }// 'strat2' is destroyed, all the ids should be still present in KTT
    // We can create kernel
    auto validKernelId = tuner.CreateSimpleKernel(kernelName2, definitionId);
    ASSERT_NE(validKernelId, ktt::InvalidKernelId);
    tuner.RemoveKernel(validKernelId);
    // All kernels run without a problem
    ASSERT_TRUE(tuner.Run(kernelId, {}, {}).IsValid());
    ASSERT_TRUE(tuner.Run(kernelId2, {}, {}).IsValid());
    // We can set the arguments
    // TODO must check that no message is printed
  }// 'strat' is destroyed, all the ids should be removed

  // Definition id is removed
  ASSERT_KTT_ERROR_MSG(tuner.CreateSimpleKernel(kernelName2, definitionId));
  // Kernel ids are removed
  ASSERT_KTT_ERROR_MSG(tuner.Run(kernelId, {}, {}));
  ASSERT_KTT_ERROR_MSG(tuner.Run(kernelId2, {}, {}));
  // Argument ids are removed
  auto validDefinitionId = tuner.AddKernelDefinitionFromFile(kernelName3, kernelFile, {}, {});
  ASSERT_NE(validDefinitionId, ktt::InvalidKernelDefinitionId);
  ASSERT_KTT_ERROR_MSG(tuner.SetArguments(validDefinitionId, { argumentId }));
  ASSERT_KTT_ERROR_MSG(tuner.SetArguments(validDefinitionId, { argumentId2 }));
  // Cleanup tuner
  tuner.RemoveKernelDefinition(validDefinitionId);
}

TEST_F(GarbageCollectionTests,
  strategies_with_different_ids_are_released_independently_on_each_other)
{
  ktt::KernelDefinitionId definitionId;
  ktt::KernelDefinitionId definitionId2;
  ktt::KernelId kernelId;
  ktt::KernelId kernelId2;
  ktt::ArgumentId argumentId;
  ktt::ArgumentId argumentId2;
  {
    TestStrategy strat(kttHelper, kernelFile);
    {
      TestStrategy strat2(kttHelper, kernelFile);
      // Strategy uses 1 definition id and 1 kernel id
      ASSERT_TRUE(strat.Init(kernelName1));
      // Strategy uses 1 argument id
      ASSERT_TRUE(strat.Execute());
      // Execute returning true and passing the assert means that all the used ids are valid
      // We can additionally check it
      definitionId = strat.testDefinitionIds.back();
      kernelId = strat.testKernelIds.back();
      argumentId = strat.testArgumentIds.back();
      ASSERT_NE(definitionId, ktt::InvalidKernelDefinitionId);
      ASSERT_NE(kernelId, ktt::InvalidKernelId);
      ASSERT_NE(argumentId, ktt::InvalidArgumentId);
      // Initialize second strat with DIFFERENT kernel
      ASSERT_TRUE(strat2.Init(kernelName2));
      ASSERT_TRUE(strat2.Execute());
      definitionId2 = strat2.testDefinitionIds.back();
      kernelId2 = strat2.testKernelIds.back();
      argumentId2 = strat2.testArgumentIds.back();
      ASSERT_NE(definitionId2, ktt::InvalidKernelDefinitionId);
      ASSERT_NE(kernelId2, ktt::InvalidKernelId);
      ASSERT_NE(argumentId2, ktt::InvalidArgumentId);
      // Check that definition ids are really different
      ASSERT_NE(definitionId, definitionId2);
    }// 'strat2' is destroyed, all the ids of the 'strat2' should be removed

    // Definition id is removed
    ASSERT_KTT_ERROR_MSG(tuner.CreateSimpleKernel(kernelName3, definitionId2));
    // Kernel id is removed
    ASSERT_KTT_ERROR_MSG(tuner.Run(kernelId2, {}, {}));
    // Argument id is removed
    auto validDefinitionId = tuner.AddKernelDefinitionFromFile(kernelName3, kernelFile, {}, {});
    ASSERT_NE(validDefinitionId, ktt::InvalidKernelDefinitionId);
    ASSERT_KTT_ERROR_MSG(tuner.SetArguments(validDefinitionId, { argumentId2 }));
    // Cleanup tuner
    tuner.RemoveKernelDefinition(validDefinitionId);
  }// 'strat' is destroyed, all the ids should be removed

  // Definition id is removed
  ASSERT_KTT_ERROR_MSG(tuner.CreateSimpleKernel(kernelName2, definitionId));
  ASSERT_KTT_ERROR_MSG(tuner.CreateSimpleKernel(kernelName2, definitionId2));
  // Kernel ids are removed
  ASSERT_KTT_ERROR_MSG(tuner.Run(kernelId, {}, {}));
  ASSERT_KTT_ERROR_MSG(tuner.Run(kernelId2, {}, {}));
  // Argument ids are removed
  auto validDefinitionId = tuner.AddKernelDefinitionFromFile(kernelName3, kernelFile, {}, {});
  ASSERT_NE(validDefinitionId, ktt::InvalidKernelDefinitionId);
  ASSERT_KTT_ERROR_MSG(tuner.SetArguments(validDefinitionId, { argumentId }));
  ASSERT_KTT_ERROR_MSG(tuner.SetArguments(validDefinitionId, { argumentId2 }));
  // Cleanup tuner
  tuner.RemoveKernelDefinition(validDefinitionId);
}

TEST_F(GarbageCollectionTests, multithreaded_test)
{
  const size_t kIterations = 1000;
  const size_t kThreads = 4;
  const std::string kKernelNames[] = { kernelName1, kernelName2, kernelName3 };

  // To silent the kernel launches
  tuner.SetLoggingLevel(ktt::LoggingLevel::Warning);

  // All the resources of this strategy should be released at the end!
  auto longlivingStrategy = std::make_unique<TestStrategy>(kttHelper, kernelFile);
  ASSERT_TRUE(longlivingStrategy->Init(kernelName1));
  ASSERT_TRUE(longlivingStrategy->Execute());
  spdlog::debug(
    "Long-living strategy definitionId: {0}", longlivingStrategy->testDefinitionIds.back());

  auto f = [this, &kKernelNames]() {
    const size_t stratCount = 5;

    // To allow strategies to live longer than one iteration
    std::unique_ptr<TestStrategy> strategyBuffer[stratCount];

    for (size_t i = 0; i < kIterations; ++i) {
      auto idx = i % stratCount;
      std::string kernelName = kKernelNames[i % 3];

      strategyBuffer[idx] = std::make_unique<TestStrategy>(kttHelper, kernelFile);

      ASSERT_TRUE(strategyBuffer[idx]->Init(kernelName));
      ASSERT_TRUE(strategyBuffer[idx]->Execute());
      // Execute returning true and passing the assert means that all the used ids are valid
    }
  };

  std::vector<std::thread> threads;

  for (size_t i = 0; i < kThreads; ++i) { threads.emplace_back(f); }

  for (auto &t : threads) { t.join(); }

  // We need copies because we want to check them after destruction of the long living strategy
  auto copyDefinitions = longlivingStrategy->testDefinitionIds;
  auto copyKernels = longlivingStrategy->testKernelIds;
  auto copyArguments = longlivingStrategy->testArgumentIds;

  // all ids of long living strategy should exist
  // We can create kernels
  for (auto definitionId : copyDefinitions) {
    auto validKernelId = tuner.CreateSimpleKernel(kernelName2, definitionId);
    ASSERT_NE(validKernelId, ktt::InvalidKernelId);
    tuner.RemoveKernel(validKernelId);
  }
  // All kernels run without a problem
  for (auto kernelId : copyKernels) { ASSERT_TRUE(tuner.Run(kernelId, {}, {}).IsValid()); }
  // All arguments can still be used
  // for (auto argumentId : copyArguments) {
  //   // We can set the arguments
  //   // TODO must check that no message is printed
  // }

  // MANUALLY destroy the long living strategy by calling reset on a unique_ptr
  longlivingStrategy.reset();

  // all ids of long living strategy should be removed
  // Check definitions
  for (auto definitionId : copyDefinitions) {
    ASSERT_KTT_ERROR_MSG(tuner.CreateSimpleKernel(kernelName2, definitionId));
  }
  // Check kernels
  for (auto kernelId : copyKernels) { ASSERT_KTT_ERROR_MSG(tuner.Run(kernelId, {}, {})); }

  // Check arguments
  auto validDefinitionId = tuner.AddKernelDefinitionFromFile(kernelName3, kernelFile, {}, {});
  for (auto argumentId : copyArguments) {
    ASSERT_NE(validDefinitionId, ktt::InvalidKernelDefinitionId);
    ASSERT_KTT_ERROR_MSG(tuner.SetArguments(validDefinitionId, { argumentId }));
  }
  tuner.RemoveKernelDefinition(validDefinitionId);

  // To restore usual logging
  tuner.SetLoggingLevel(ktt::LoggingLevel::Info);
}
