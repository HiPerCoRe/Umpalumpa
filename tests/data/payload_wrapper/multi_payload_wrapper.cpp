#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <libumpalumpa/data/multi_payload_wrapper.hpp>

using namespace umpalumpa::data;

// TODO proper mocks and tests

struct BasePayload
{
  BasePayload() {}
  BasePayload(int data) : data(new int(data)) {}

  virtual bool IsEquivalentTo(const BasePayload &) const = 0;
  std::unique_ptr<int> data;
};

struct AllTruePayload : public BasePayload
{
  using BasePayload::BasePayload;
  AllTruePayload CopyWithoutData() const { return AllTruePayload(); }
  bool IsValid() const { return true; }
  bool IsEquivalentTo(const BasePayload &) const override { return true; }
};

struct AllFalsePayload : public BasePayload
{
  using BasePayload::BasePayload;
  AllFalsePayload CopyWithoutData() const { return AllFalsePayload(); }
  bool IsValid() const { return false; }
  bool IsEquivalentTo(const BasePayload &) const override { return false; }
};

template<typename... Args> struct TestPayloadWrapper : public MultiPayloadWrapper<Args...>
{
  TestPayloadWrapper(Args &&... args) : MultiPayloadWrapper<Args...>(std::move(args)...) {}

  const auto &GetPayloads() { return MultiPayloadWrapper<Args...>::payloads; }
  TestPayloadWrapper CopyWithoutData() const
  {
    return MultiPayloadWrapper<Args...>::template CopyWithoutData<TestPayloadWrapper>();
  }
};

// NOTE currently not supported, might not be needed
// TEST(TestPayloadWrapperTest, Create_MPW_with_multiple_lvalue_payloads_same_type)
// {
//   // This tests mainly the ability to compile the code using TestPayloadWrapper
//   auto p1 = AllTruePayload();
//   auto p2 = AllTruePayload();
//   auto p3 = AllTruePayload();
//   auto mpw = TestPayloadWrapper(p1, p2, p3);
// }

TEST(TestPayloadWrapperTest, Create_MPW_with_multiple_rvalue_payloads_same_type)
{
  // This tests mainly the ability to compile the code using TestPayloadWrapper
  auto mpw = TestPayloadWrapper(AllTruePayload(), AllTruePayload(), AllTruePayload());
}

TEST(TestPayloadWrapperTest, Create_MPW_with_multiple_rvalue_payloads_various_types)
{
  // This tests mainly the ability to compile the code using TestPayloadWrapper
  auto mpw =
    TestPayloadWrapper(AllTruePayload(), AllFalsePayload(), AllFalsePayload(), AllTruePayload());
}

TEST(TestPayloadWrapperTest, Get_with_value_present)
{
  const int val0 = 10;
  const int val1 = 42;
  auto mpw = TestPayloadWrapper(AllTruePayload(val0), AllFalsePayload(val1));

  const auto &p0 = std::get<0>(mpw.GetPayloads());
  const auto &p1 = std::get<1>(mpw.GetPayloads());

  ASSERT_NE(p0.data.get(), nullptr);
  ASSERT_NE(p1.data.get(), nullptr);
  ASSERT_EQ(*p0.data, val0);
  ASSERT_EQ(*p1.data, val1);
}

TEST(TestPayloadWrapperTest, Get_with_value_missing)
{
  auto mpw = TestPayloadWrapper(AllTruePayload(), AllFalsePayload());

  const auto &p0 = std::get<0>(mpw.GetPayloads());
  const auto &p1 = std::get<1>(mpw.GetPayloads());

  ASSERT_EQ(p0.data.get(), nullptr);
  ASSERT_EQ(p1.data.get(), nullptr);
}

TEST(TestPayloadWrapperTest, IsValid_all_payloads_always_return_true)
{
  auto mpw = TestPayloadWrapper(AllTruePayload(), AllTruePayload(), AllTruePayload());
  ASSERT_TRUE(mpw.IsValid());
}

TEST(TestPayloadWrapperTest, IsValid_all_payloads_always_return_false)
{
  auto mpw = TestPayloadWrapper(AllFalsePayload(), AllFalsePayload(), AllFalsePayload());
  ASSERT_FALSE(mpw.IsValid());
}

TEST(TestPayloadWrapperTest, IsValid_one_payload_returns_false_rest_return_true)
{
  auto mpw = TestPayloadWrapper(AllTruePayload(), AllFalsePayload(), AllTruePayload());
  ASSERT_FALSE(mpw.IsValid());
}

TEST(TestPayloadWrapperTest, IsEquivalentTo_all_payloads_always_return_true)
{
  auto mpw = TestPayloadWrapper(AllTruePayload(), AllTruePayload(), AllTruePayload());
  auto mpw2 = TestPayloadWrapper(AllTruePayload(), AllTruePayload(), AllTruePayload());
  ASSERT_TRUE(mpw.IsEquivalentTo(mpw2));
}

TEST(TestPayloadWrapperTest, IsEquivalentTo_all_payloads_always_return_false)
{
  auto mpw = TestPayloadWrapper(AllFalsePayload(), AllFalsePayload(), AllFalsePayload());
  auto mpw2 = TestPayloadWrapper(AllFalsePayload(), AllFalsePayload(), AllFalsePayload());
  ASSERT_FALSE(mpw.IsEquivalentTo(mpw2));
}

TEST(TestPayloadWrapperTest, IsEquivalentTo_one_payload_returns_false_rest_return_true)
{
  auto mpw = TestPayloadWrapper(AllTruePayload(), AllFalsePayload(), AllTruePayload());
  auto mpw2 = TestPayloadWrapper(AllTruePayload(), AllFalsePayload(), AllTruePayload());
  ASSERT_FALSE(mpw.IsEquivalentTo(mpw2));
}

// NOTE Can't be equivalent when they have different types, compile-time check
// TEST(TestPayloadWrapperTest, IsEquivalentTo_different_types)
// {
//   auto mpw = TestPayloadWrapper(AllTruePayload(), AllTruePayload());
//   auto mpw2 = TestPayloadWrapper(AllTruePayload(), AllFalsePayload());
//   ASSERT_FALSE(mpw.IsEquivalentTo(mpw2));
// }

// NOTE Can't be equivalent when they have different length, compile-time check
// TEST(TestPayloadWrapperTest, IsEquivalentTo_different_length)
// {
//   auto mpw = TestPayloadWrapper(AllTruePayload(), AllTruePayload());
//   auto mpw2 = TestPayloadWrapper(AllTruePayload(), AllTruePayload(), AllTruePayload());
//   ASSERT_FALSE(mpw.IsEquivalentTo(mpw2));
// }

TEST(TestPayloadWrapperTest, CopyWithoutData_no_data)
{
  auto mpw = TestPayloadWrapper(AllTruePayload(), AllFalsePayload());

  auto noDataCopy = mpw.CopyWithoutData();
  const auto &p0 = std::get<0>(noDataCopy.GetPayloads());
  const auto &p1 = std::get<1>(noDataCopy.GetPayloads());

  ASSERT_EQ(p0.data.get(), nullptr);
  ASSERT_EQ(p1.data.get(), nullptr);
}

TEST(TestPayloadWrapperTest, CopyWithoutData_original_payloads_have_data)
{
  const int val0 = 10;
  const int val1 = 42;
  auto mpw = TestPayloadWrapper(AllTruePayload(val0), AllFalsePayload(val1));

  auto noDataCopy = mpw.CopyWithoutData();
  const auto &noDataP0 = std::get<0>(noDataCopy.GetPayloads());
  const auto &noDataP1 = std::get<1>(noDataCopy.GetPayloads());
  const auto &p0 = std::get<0>(mpw.GetPayloads());
  const auto &p1 = std::get<1>(mpw.GetPayloads());

  ASSERT_EQ(noDataP0.data.get(), nullptr);
  ASSERT_EQ(noDataP1.data.get(), nullptr);

  ASSERT_NE(p0.data.get(), nullptr);
  ASSERT_NE(p1.data.get(), nullptr);
  ASSERT_EQ(*p0.data, val0);
  ASSERT_EQ(*p1.data, val1);
}
