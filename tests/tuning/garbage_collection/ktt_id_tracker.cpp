#include <gtest/gtest.h>
#include <libumpalumpa/tuning/ktt_helper.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>
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
    ASSERT_TRUE(StartsWith(output, "[Error]")) << "Output is: '" << output << "'";}
// clang-format on

// TESTS FOR KTTIdTracker
// Not thread safe on its own!! by design

class KTTIdTrackerTests : public Test
{
protected:
  KTTIdTrackerTests() : baseAlgo(0), kttHelper(baseAlgo.GetHelper()), tuner(kttHelper.GetTuner()) {}

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


TEST_F(KTTIdTrackerTests, when_destroyed_properly_releases_definitionId)
{
  auto definitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  {
    auto tracker = kttHelper.GetIdTracker(definitionId);
    // Check that definition id is in KTT by creating a valid kernel
    auto kernelId = tuner.CreateSimpleKernel(kernelName1, definitionId);
    ASSERT_NE(kernelId, ktt::InvalidKernelId);
    // Remove id again to not affect the rest of the test
    tuner.RemoveKernel(kernelId);

    kttHelper.CleanupIdTracker(tracker);
  }// tracker destroyed -> id released from tuner

  // Check removal of ids by attempting to create a kernel
  ASSERT_KTT_ERROR_MSG(auto kernelId = tuner.CreateSimpleKernel(kernelName1, definitionId));
  ASSERT_EQ(kernelId, ktt::InvalidKernelId);
}

TEST_F(KTTIdTrackerTests, when_destroyed_properly_releases_kernelIds)
{
  auto definitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  {
    auto tracker = kttHelper.GetIdTracker(definitionId);
    // Check that definition id is in KTT by creating a valid kernel
    auto kernelId = tuner.CreateSimpleKernel(kernelName1, definitionId);
    ASSERT_NE(kernelId, ktt::InvalidKernelId);
    // Add kernel id into the tracker
    tracker->kernelIds.push_back(kernelId);

    kttHelper.CleanupIdTracker(tracker);
  }// tracker destroyed -> ids released from tuner

  // Check removal of ids by attempting to create a kernel
  ASSERT_KTT_ERROR_MSG(auto kernelId = tuner.CreateSimpleKernel(kernelName1, definitionId));
  ASSERT_EQ(kernelId, ktt::InvalidKernelId);
}

TEST_F(KTTIdTrackerTests, when_destroyed_properly_releases_argumentIds)
{
  auto definitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  ktt::ArgumentId argumentId;
  {
    auto tracker = kttHelper.GetIdTracker(definitionId);
    // Check that definition id is in KTT by creating a valid kernel
    auto kernelId = tuner.CreateSimpleKernel(kernelName1, definitionId);
    ASSERT_NE(kernelId, ktt::InvalidKernelId);
    // Add kernel id into the tracker
    tracker->kernelIds.push_back(kernelId);
    argumentId = tuner.AddArgumentScalar(NULL);
    tuner.SetArguments(definitionId, { argumentId });
    // Add argument id into the tracker
    tracker->argumentIds.push_back(argumentId);
    // To check that argument id is really in the tuner we run a kernel
    ASSERT_TRUE(tuner.Run(kernelId, {}, {}).IsValid());

    kttHelper.CleanupIdTracker(tracker);
  }// tracker destroyed -> ids released from tuner

  // Check removal of ids by attempting to create a kernel
  ASSERT_KTT_ERROR_MSG(auto kernelId = tuner.CreateSimpleKernel(kernelName1, definitionId));
  ASSERT_EQ(kernelId, ktt::InvalidKernelId);

  // To check that argument id is really removed from KTT, we try to associate it with definition id
  auto checkDefinitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  ASSERT_KTT_ERROR_MSG(tuner.SetArguments(checkDefinitionId, { argumentId }));

  // Clean KTT because we are dealing with global state
  tuner.RemoveKernelDefinition(checkDefinitionId);
}

TEST_F(KTTIdTrackerTests, tracker_destroyed_only_after_losing_last_reference_to_it)
{
  auto definitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  {
    auto tracker1 = kttHelper.GetIdTracker(definitionId);
    {
      auto tracker2 = kttHelper.GetIdTracker(definitionId);
      {
        auto tracker3 = kttHelper.GetIdTracker(definitionId);
        // Check that definition id is in KTT by creating a valid kernel
        auto kernelId3 = tuner.CreateSimpleKernel(kernelName3, definitionId);
        ASSERT_NE(kernelId3, ktt::InvalidKernelId);
        // Remove id again to not affect the rest of the test
        tuner.RemoveKernel(kernelId3);
      }// 2 references left

      // Check that definition id is in KTT by creating a valid kernel
      auto kernelId2 = tuner.CreateSimpleKernel(kernelName2, definitionId);
      ASSERT_NE(kernelId2, ktt::InvalidKernelId);
      // Remove id again to not affect the rest of the test
      tuner.RemoveKernel(kernelId2);
    }// 1 reference left

    // Check that definition id is in KTT by creating a valid kernel
    auto kernelId1 = tuner.CreateSimpleKernel(kernelName1, definitionId);
    ASSERT_NE(kernelId1, ktt::InvalidKernelId);
    // Remove id again to not affect the rest of the test
    tuner.RemoveKernel(kernelId1);

    kttHelper.CleanupIdTracker(tracker1);
  }// tracker destroyed -> ids released from the tuner

  // Check removal of ids by attempting to create a kernel
  ASSERT_KTT_ERROR_MSG(auto kernelId = tuner.CreateSimpleKernel(kernelName1, definitionId));
  ASSERT_EQ(kernelId, ktt::InvalidKernelId);
}

TEST_F(KTTIdTrackerTests, multithreaded_test)
{
  const size_t kIterations = 1000;
  const size_t kThreads = 4;
  const std::string kKernelNames[] = { kernelName1, kernelName2, kernelName3 };

  // This definition id should be released at the very end, NOT sooner
  auto definitionId = tuner.AddKernelDefinitionFromFile(kernelName1, kernelFile, {}, {});
  // This tracker should be destroyed at the very end, NOT sooner
  auto tracker = kttHelper.GetIdTracker(definitionId);

  std::vector<std::weak_ptr<KTTIdTracker>> checkExpired;
  std::vector<std::weak_ptr<KTTIdTracker>> checkNotExpired;

  auto f = [this, &kKernelNames, &checkExpired, &checkNotExpired]() {
    std::stringstream ss;
    ss << std::this_thread::get_id();
    const std::string prefix = ss.str() + "_";

    std::shared_ptr<KTTIdTracker> trackers[3];

    for (size_t i = 0; i < kIterations; ++i) {
      auto idx = i % 3;
      std::string kernelName = kKernelNames[idx];
      std::string finalKernelName = prefix + kernelName + "_" + std::to_string(i);
      std::shared_ptr<KTTIdTracker> &currentTracker = trackers[idx];

      {// Simulate strategy Init
        std::lock_guard lck(kttHelper.GetMutex());
        // Acquire some KTTIdTracker (one or more)
        // Add definitionId
        auto dId = tuner.GetKernelDefinitionId(kernelName);
        if (dId == ktt::InvalidKernelDefinitionId) {
          dId = tuner.AddKernelDefinitionFromFile(kernelName, kernelFile, {}, {});
        }
        currentTracker = kttHelper.GetIdTracker(dId);
        if (idx > 0) {
          checkExpired.emplace_back(currentTracker);
        } else {
          checkNotExpired.emplace_back(currentTracker);
        }

        // Add kernelId
        auto kId = tuner.CreateSimpleKernel(finalKernelName, dId);
        currentTracker->kernelIds.push_back(kId);
      }
      {// Simulate strategy Execute
        std::lock_guard lck(kttHelper.GetMutex());
        // Add argumentId
        auto aId = tuner.AddArgumentScalar(NULL);
        tuner.SetArguments(currentTracker->definitionId, { aId });
        currentTracker->argumentIds.push_back(aId);
      }
      {// Simulate strategy Destructor
        std::lock_guard lck(kttHelper.GetMutex());
        kttHelper.CleanupIdTracker(currentTracker);
      }
    }
  };

  std::vector<std::thread> threads;

  for (size_t i = 0; i < kThreads; ++i) { threads.emplace_back(f); }

  for (auto &t : threads) { t.join(); }

  // All the other trackers should be destroyed by now
  for (auto &checkTracker : checkExpired) { ASSERT_TRUE(checkTracker.expired()); }
  for (auto &checkTracker : checkNotExpired) { ASSERT_FALSE(checkTracker.expired()); }

  // tracker should exist
  std::weak_ptr checkTracker(tracker);
  ASSERT_FALSE(checkTracker.expired());
  // definition id should exist, create kernel to check
  auto validKernelId = tuner.CreateSimpleKernel("Valid", definitionId);
  ASSERT_NE(validKernelId, ktt::InvalidKernelId);
  tuner.RemoveKernel(validKernelId);

  kttHelper.CleanupIdTracker(tracker);

  ASSERT_TRUE(checkTracker.expired());
  ASSERT_KTT_ERROR_MSG(auto invalidKernelId = tuner.CreateSimpleKernel("Invalid", definitionId));
  ASSERT_EQ(invalidKernelId, ktt::InvalidKernelId);
}
