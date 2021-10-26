#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <libumpalumpa/data/multi_payload_wrapper.hpp>

using namespace umpalumpa::data;

// TODO proper mocks and tests

struct AllTruePayload
{
  AllTruePayload CopyWithoutData() const { return AllTruePayload(); }
  bool IsValid() const { return true; }
  bool IsEquivalentTo(const AllTruePayload &) const { return true; }
};

class MultiPayloadWrapperTest : public ::testing::Test
{
  // TODO
};

TEST_F(MultiPayloadWrapperTest, Create) { auto mpw = MultiPayloadWrapper(AllTruePayload()); }

TEST_F(MultiPayloadWrapperTest, Create2)
{
  auto mpw = MultiPayloadWrapper(AllTruePayload(), AllTruePayload());
}

TEST_F(MultiPayloadWrapperTest, Get)
{
  auto mpw = MultiPayloadWrapper(AllTruePayload(), AllTruePayload());
  auto p1 = std::get<1>(mpw.payload);
}

TEST_F(MultiPayloadWrapperTest, IsValid)
{
  auto mpw = MultiPayloadWrapper(AllTruePayload(), AllTruePayload());
  ASSERT_TRUE(mpw.IsValid());
}

TEST_F(MultiPayloadWrapperTest, IsEquivalentTo)
{
  auto mpw = MultiPayloadWrapper(AllTruePayload(), AllTruePayload());
  auto mpw2 = MultiPayloadWrapper(AllTruePayload(), AllTruePayload());
  ASSERT_TRUE(mpw.IsEquivalentTo(mpw2));
}
