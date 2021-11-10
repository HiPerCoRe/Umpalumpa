#include <gtest/gtest.h>
#include <iostream>

#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/garbage_collector.hpp>

using namespace ::testing;
using namespace umpalumpa;
using namespace umpalumpa::utils;
using namespace umpalumpa::algorithm;

class GarbageCollectorTests : public Test
{
protected:
  GarbageCollectorTests()
    : baseAlgo(0), kttHelper(baseAlgo.GetHelper()), tuner(kttHelper.GetTuner())
  {}

  KTT_Base baseAlgo;
  KTTHelper &kttHelper;
  ktt::Tuner &tuner;

  static const inline std::string kernelFile =
    utils::GetSourceFilePath("tests/tuning/garbage_collector/test_kernels.cu");
  static const inline std::string kernelName1 = "TestKernel1";
  static const inline std::string kernelName2 = "TestKernel2";
  static const inline std::string kernelName3 = "TestKernel3";

  bool StartsWith(const std::string &s, const std::string &prefix) const
  {
    return s.rfind(prefix, 0) == 0;
  }
  void PreExpectedErrorMsg() { internal::CaptureStderr(); }
  void PostExpectedErrorMsg()
  {
    auto output = internal::GetCapturedStderr();
    ASSERT_TRUE(StartsWith(output, "[ERROR]"));
  }
};

TEST_F(GarbageCollectorTests, removal_of_definitionId_from_KTT_using_GarbageCollector)
{
  auto definitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  // Definition id is saved in KTT

  // To test that id is really in KTT we create a KTT kernel
  auto kernelId = tuner.CreateSimpleKernel("Test", definitionId);
  ASSERT_NE(kernelId, ktt::InvalidKernelId);
  // We remove the kernel to not affect the rest of the test
  tuner.RemoveKernel(kernelId);

  GarbageCollector gc;
  gc.RegisterKernelDefinitionId(definitionId, kttHelper);

  gc.CleanupIds(kttHelper, {}, { definitionId });
  // Definition id is removed from the KTT

  // To test that id is really removed, we try to create a KTT kernel
  PreExpectedErrorMsg();
  auto invalidKernelId = tuner.CreateSimpleKernel("Invalid", definitionId);
  PostExpectedErrorMsg();

  ASSERT_EQ(invalidKernelId, ktt::InvalidKernelId);
}

TEST_F(GarbageCollectorTests, removal_of_several_definitionIds_from_KTT_using_GarbageCollector)
{
  auto definitionId1 = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  auto definitionId2 = tuner.AddKernelDefinitionFromFile(kernelName2, kernelFile, {}, {});
  auto definitionId3 = tuner.AddKernelDefinitionFromFile(kernelName3, kernelFile, {}, {});
  // Definition id is saved in KTT

  // To test that id is really in KTT we create a KTT kernel
  auto kernelId1 = tuner.CreateSimpleKernel(kernelName1, definitionId1);
  auto kernelId2 = tuner.CreateSimpleKernel(kernelName2, definitionId2);
  auto kernelId3 = tuner.CreateSimpleKernel(kernelName3, definitionId3);
  ASSERT_NE(kernelId1, ktt::InvalidKernelId);
  ASSERT_NE(kernelId2, ktt::InvalidKernelId);
  ASSERT_NE(kernelId3, ktt::InvalidKernelId);
  // We remove the kernel to not affect the rest of the test
  tuner.RemoveKernel(kernelId1);
  tuner.RemoveKernel(kernelId2);
  tuner.RemoveKernel(kernelId3);

  GarbageCollector gc;
  gc.RegisterKernelDefinitionId(definitionId1, kttHelper);
  gc.RegisterKernelDefinitionId(definitionId2, kttHelper);
  gc.RegisterKernelDefinitionId(definitionId3, kttHelper);

  gc.CleanupIds(kttHelper, {}, { definitionId1, definitionId2, definitionId3 });
  // Definition id is removed from the KTT

  // To test that id is really removed, we try to create a KTT kernel
  PreExpectedErrorMsg();
  auto invalidKernelId1 = tuner.CreateSimpleKernel("Invalid1", definitionId1);
  PostExpectedErrorMsg();
  PreExpectedErrorMsg();
  auto invalidKernelId2 = tuner.CreateSimpleKernel("Invalid2", definitionId2);
  PostExpectedErrorMsg();
  PreExpectedErrorMsg();
  auto invalidKernelId3 = tuner.CreateSimpleKernel("Invalid3", definitionId3);
  PostExpectedErrorMsg();

  ASSERT_EQ(invalidKernelId1, ktt::InvalidKernelId);
  ASSERT_EQ(invalidKernelId2, ktt::InvalidKernelId);
  ASSERT_EQ(invalidKernelId3, ktt::InvalidKernelId);
}

TEST_F(GarbageCollectorTests, removal_of_kernelId_from_KTT_using_GarbageCollector)
{
  auto definitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  // Definition id is saved in KTT

  // To test that id is really in KTT we create a KTT kernel
  auto kernelId = tuner.CreateSimpleKernel("Test", definitionId);
  ASSERT_NE(kernelId, ktt::InvalidKernelId);

  GarbageCollector gc;
  gc.RegisterKernelDefinitionId(definitionId, kttHelper);

  gc.CleanupIds(kttHelper, { kernelId }, { definitionId });

  // To test that id is really removed, we try to create a KTT kernel
  PreExpectedErrorMsg();
  auto invalidKernelId = tuner.CreateSimpleKernel("Invalid", definitionId);
  PostExpectedErrorMsg();
  // Definition id could only be removed when kernel id was removed before

  ASSERT_EQ(invalidKernelId, ktt::InvalidKernelId);
}

TEST_F(GarbageCollectorTests, more_kernelIds_for_one_definitionId)
{
  auto definitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  // Definition id is saved in KTT

  auto kernelId1 = tuner.CreateSimpleKernel("Test1", definitionId);
  auto kernelId2 = tuner.CreateSimpleKernel("Test2", definitionId);
  ASSERT_NE(kernelId1, ktt::InvalidKernelId);
  ASSERT_NE(kernelId2, ktt::InvalidKernelId);

  GarbageCollector gc;
  gc.RegisterKernelDefinitionId(definitionId, kttHelper);

  gc.CleanupIds(kttHelper, { kernelId1, kernelId2 }, { definitionId });

  // To test that id is really removed, we try to create a KTT kernel
  PreExpectedErrorMsg();
  auto invalidKernelId = tuner.CreateSimpleKernel("Invalid", definitionId);
  PostExpectedErrorMsg();
  // Definition id could only be removed when kernel id was removed before

  ASSERT_EQ(invalidKernelId, ktt::InvalidKernelId);
}

TEST_F(GarbageCollectorTests, definitionId_removed_only_when_no_strategy_uses_it)
{
  auto definitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  // Definition id is saved in KTT

  // Kernel of strategy1
  auto kernelId1 = tuner.CreateSimpleKernel("Test1", definitionId);
  // Kernel of strategy2
  auto kernelId2 = tuner.CreateSimpleKernel("Test2", definitionId);

  ASSERT_NE(kernelId1, ktt::InvalidKernelId);
  ASSERT_NE(kernelId2, ktt::InvalidKernelId);

  GarbageCollector gc;
  // Called in Init of strategy1
  gc.RegisterKernelDefinitionId(definitionId, kttHelper);
  // Called in Init of strategy2
  gc.RegisterKernelDefinitionId(definitionId, kttHelper);

  // strategy1 is destroyed
  gc.CleanupIds(kttHelper, { kernelId1 }, { definitionId });

  // Definition id is not removed because of existence of strategy2
  auto validKernelId = tuner.CreateSimpleKernel("Valid", definitionId);

  ASSERT_NE(validKernelId, ktt::InvalidKernelId);

  // Clean up KTT, needed because we are dealing with global state
  tuner.RemoveKernel(kernelId2);
  tuner.RemoveKernel(validKernelId);
  tuner.RemoveKernelDefinition(definitionId);
}

TEST_F(GarbageCollectorTests, removal_of_argumentId_from_KTT_using_GarbageCollector)
{
  auto definitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  // Definition id is saved in KTT

  // To test that id is really in KTT we create a KTT kernel
  auto kernelId = tuner.CreateSimpleKernel("Test", definitionId);
  ASSERT_NE(kernelId, ktt::InvalidKernelId);

  auto argumentId = tuner.AddArgumentScalar(NULL);
  // To test that argumentId is really in KTT we run the kernel
  tuner.SetArguments(definitionId, { argumentId });
  ASSERT_TRUE(tuner.Run(kernelId, {}, {}).IsValid());

  GarbageCollector gc;
  gc.RegisterKernelDefinitionId(definitionId, kttHelper);
  gc.RegisterArgumentIds(definitionId, { argumentId }, kttHelper);

  gc.CleanupIds(kttHelper, { kernelId }, { definitionId });

  auto definitionId2 = tuner.AddKernelDefinitionFromFile(kernelName2, kernelFile, {}, {});
  // To test that argument id is really removed, we try to associate it with some definition id
  PreExpectedErrorMsg();
  tuner.SetArguments(definitionId2, { argumentId });
  PostExpectedErrorMsg();
}

// FIXME this test can only run when we can create more instances of KTTHelper
// change this condition to something reasonable
#if GPU_COUNT > 1
TEST_F(GarbageCollectorTests,
  removal_of_definitionId_from_KTT_using_GarbageCollector_while_having_more_KTTs)
{
  KTT_Base ktt_base2(1);
  auto &kttHelper2 = ktt_base2.GetHelper();
  auto &tuner2 = kttHelper2.GetTuner();

  auto definitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  auto definitionId2 = tuner2.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});

  // To test that id is really in KTT we create a KTT kernel
  auto kernelId = tuner.CreateSimpleKernel("Test", definitionId);
  auto kernelId2 = tuner2.CreateSimpleKernel("Test", definitionId2);
  ASSERT_NE(kernelId, ktt::InvalidKernelId);
  ASSERT_NE(kernelId2, ktt::InvalidKernelId);
  // We remove the kernel to not affect the rest of the test
  tuner.RemoveKernel(kernelId);
  tuner2.RemoveKernel(kernelId2);

  GarbageCollector gc;
  gc.RegisterKernelDefinitionId(definitionId, kttHelper);
  gc.RegisterKernelDefinitionId(definitionId2, kttHelper2);

  gc.CleanupIds(kttHelper, {}, { definitionId });
  // Only definition id of kttHelper should be removed
  // kttHelper2 should be untouched

  PreExpectedErrorMsg();
  auto invalidKernelId = tuner.CreateSimpleKernel("Invalid", definitionId);
  PostExpectedErrorMsg();
  auto validKernelId2 = tuner2.CreateSimpleKernel("Valid2", definitionId2);

  ASSERT_EQ(invalidKernelId, ktt::InvalidKernelId);
  ASSERT_NE(validKernelId2, ktt::InvalidKernelId);

  // Clean KTT2 at the end, because we are dealing with global state
  tuner2.RemoveKernel(validKernelId2);
  tuner2.RemoveKernelDefinition(definitionId2);
}
#endif

