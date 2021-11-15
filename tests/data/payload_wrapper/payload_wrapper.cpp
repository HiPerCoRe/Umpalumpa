#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <libumpalumpa/data/payload_wrapper.hpp>

using namespace ::testing;
using namespace umpalumpa::data;

struct MockPayload
{
  MOCK_METHOD(bool, IsEquivalentTo, (const MockPayload &), (const));
  MOCK_METHOD(bool, IsValid, (), (const));
  MOCK_METHOD(std::shared_ptr<MockPayload>, CopyWithoutData, (), (const));
  MOCK_METHOD(int, GetData, (), (const));
};

// Mocked object of gmock framework can't be copied or moved
struct MockPayloadWrapper
{
  MockPayloadWrapper() = default;
  MockPayloadWrapper(const std::shared_ptr<MockPayload> &mp) : mock(mp) {}

  bool IsEquivalentTo(const MockPayloadWrapper &mp) const { return mock->IsEquivalentTo(*mp.mock); }
  bool IsValid() const { return mock->IsValid(); }
  MockPayloadWrapper CopyWithoutData() const { return MockPayloadWrapper(mock->CopyWithoutData()); }
  int GetData() const { return mock->GetData(); }

  std::shared_ptr<MockPayload> mock = std::make_shared<MockPayload>();

  // ugly way to check that new instance has been created...
  const size_t testId = idCounter++;

private:
  inline static size_t idCounter = 0;
};

// Define another type to test multiple types
struct MockPayloadWrapperOther : public MockPayloadWrapper
{
};

template<typename... Args> struct TestPayloadWrapper : public PayloadWrapper<Args...>
{
  TestPayloadWrapper(std::tuple<Args...> &t) : PayloadWrapper<Args...>(t) {}
  TestPayloadWrapper(Args &...args) : PayloadWrapper<Args...>(args...) {}

  const auto &GetPayloads() { return PayloadWrapper<Args...>::payloads; }
};

TEST(TestPayloadWrapperTest, Create_MPW_with_multiple_rvalue_payloads_same_type)
{
  // This tests mainly the ability to compile the code using TestPayloadWrapper
  auto mpw1 = MockPayloadWrapper();
  auto mpw2 = MockPayloadWrapper();
  auto mpw3 = MockPayloadWrapper();
  auto tpw = TestPayloadWrapper(mpw1, mpw2, mpw3);
}

TEST(TestPayloadWrapperTest, Create_MPW_with_multiple_rvalue_payloads_various_types)
{
  auto mpw1 = MockPayloadWrapper();
  auto mpw2 = MockPayloadWrapper();
  auto mpwo1 = MockPayloadWrapperOther();
  auto mpwo2 = MockPayloadWrapperOther();
  // This tests mainly the ability to compile the code using TestPayloadWrapper
  auto mpw = TestPayloadWrapper(mpw1, mpwo1, mpwo2, mpw2);
}

TEST(TestPayloadWrapperTest, Get_payload)
{
  const int val0 = 10;
  const int val1 = 42;
  MockPayloadWrapper m0;
  MockPayloadWrapper m1;
  EXPECT_CALL(*m0.mock, GetData()).WillRepeatedly(Return(val0));
  EXPECT_CALL(*m1.mock, GetData()).WillRepeatedly(Return(val1));

  auto mpw = TestPayloadWrapper(m0, m1);

  const auto &p0 = std::get<0>(mpw.GetPayloads());
  const auto &p1 = std::get<1>(mpw.GetPayloads());

  ASSERT_EQ(p0.GetData(), val0);
  ASSERT_EQ(p1.GetData(), val1);
}

TEST(TestPayloadWrapperTest, IsValid_all_payloads_always_return_true)
{
  MockPayloadWrapper m0;
  MockPayloadWrapper m1;
  MockPayloadWrapper m2;
  EXPECT_CALL(*m0.mock, IsValid()).WillRepeatedly(Return(true));
  EXPECT_CALL(*m1.mock, IsValid()).WillRepeatedly(Return(true));
  EXPECT_CALL(*m2.mock, IsValid()).WillRepeatedly(Return(true));
  auto mpw = TestPayloadWrapper(m0, m1, m2);

  ASSERT_TRUE(mpw.IsValid());
}

TEST(TestPayloadWrapperTest, IsValid_all_payloads_always_return_false)
{
  MockPayloadWrapper m0;
  MockPayloadWrapper m1;
  MockPayloadWrapper m2;
  EXPECT_CALL(*m0.mock, IsValid()).WillRepeatedly(Return(false));
  EXPECT_CALL(*m1.mock, IsValid()).WillRepeatedly(Return(false));
  EXPECT_CALL(*m2.mock, IsValid()).WillRepeatedly(Return(false));
  auto mpw = TestPayloadWrapper(m0, m1, m2);

  ASSERT_FALSE(mpw.IsValid());
}

TEST(TestPayloadWrapperTest, IsValid_one_payload_returns_false_rest_return_true)
{
  MockPayloadWrapper m0;
  MockPayloadWrapper m1;
  MockPayloadWrapper m2;
  EXPECT_CALL(*m0.mock, IsValid()).WillRepeatedly(Return(true));
  EXPECT_CALL(*m1.mock, IsValid()).WillRepeatedly(Return(false));
  EXPECT_CALL(*m2.mock, IsValid()).WillRepeatedly(Return(true));
  auto mpw = TestPayloadWrapper(m0, m1, m2);

  ASSERT_FALSE(mpw.IsValid());
}

TEST(TestPayloadWrapperTest, IsEquivalentTo_all_payloads_always_return_true)
{
  MockPayloadWrapper m0;
  MockPayloadWrapper m1;
  MockPayloadWrapper m2;
  MockPayloadWrapper m3;
  EXPECT_CALL(*m0.mock, IsEquivalentTo).WillRepeatedly(Return(true));
  EXPECT_CALL(*m1.mock, IsEquivalentTo).WillRepeatedly(Return(true));
  EXPECT_CALL(*m2.mock, IsEquivalentTo).WillRepeatedly(Return(true));
  EXPECT_CALL(*m3.mock, IsEquivalentTo).WillRepeatedly(Return(true));
  auto mpw = TestPayloadWrapper(m0, m1);
  auto mpw2 = TestPayloadWrapper(m2, m3);

  ASSERT_TRUE(mpw.IsEquivalentTo(mpw2));
}

TEST(TestPayloadWrapperTest, IsEquivalentTo_all_payloads_always_return_false)
{
  MockPayloadWrapper m0;
  MockPayloadWrapper m1;
  MockPayloadWrapper m2;
  MockPayloadWrapper m3;
  EXPECT_CALL(*m0.mock, IsEquivalentTo).WillRepeatedly(Return(false));
  EXPECT_CALL(*m1.mock, IsEquivalentTo).WillRepeatedly(Return(false));
  EXPECT_CALL(*m2.mock, IsEquivalentTo).WillRepeatedly(Return(false));
  EXPECT_CALL(*m3.mock, IsEquivalentTo).WillRepeatedly(Return(false));
  auto mpw = TestPayloadWrapper(m0, m1);
  auto mpw2 = TestPayloadWrapper(m2, m3);

  ASSERT_FALSE(mpw.IsEquivalentTo(mpw2));
}

TEST(TestPayloadWrapperTest, IsEquivalentTo_one_payload_returns_false_rest_return_true)
{
  MockPayloadWrapper m0;
  MockPayloadWrapper m1;
  MockPayloadWrapper m2;
  MockPayloadWrapper m3;
  EXPECT_CALL(*m0.mock, IsEquivalentTo).WillRepeatedly(Return(true));
  EXPECT_CALL(*m1.mock, IsEquivalentTo).WillRepeatedly(Return(false));
  EXPECT_CALL(*m2.mock, IsEquivalentTo).WillRepeatedly(Return(true));
  EXPECT_CALL(*m3.mock, IsEquivalentTo).WillRepeatedly(Return(false));
  auto mpw = TestPayloadWrapper(m0, m1);
  auto mpw2 = TestPayloadWrapper(m2, m3);

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
  MockPayloadWrapper m0;
  MockPayloadWrapper m1;
  EXPECT_CALL(*m0.mock, CopyWithoutData()).WillOnce(Return(nullptr));
  EXPECT_CALL(*m1.mock, CopyWithoutData()).WillOnce(Return(nullptr));
  auto mpw = TestPayloadWrapper(m0, m1);

  auto payloadsNoData = mpw.CopyWithoutData();
  TestPayloadWrapper noDataCopy(payloadsNoData);
  const auto &noDataP0 = std::get<0>(noDataCopy.GetPayloads());
  const auto &noDataP1 = std::get<1>(noDataCopy.GetPayloads());
  const auto &p0 = std::get<0>(mpw.GetPayloads());
  const auto &p1 = std::get<1>(mpw.GetPayloads());

  ASSERT_EQ(noDataP0.mock.get(), nullptr);
  ASSERT_EQ(noDataP1.mock.get(), nullptr);
  // Checks that we created new instance
  ASSERT_NE(p0.testId, noDataP0.testId);
  ASSERT_NE(p1.testId, noDataP1.testId);
}

TEST(TestPayloadWrapperTest, CopyWithoutData_original_payloads_have_data)
{
  const int val0 = 10;
  const int val1 = 42;
  MockPayloadWrapper m0;
  MockPayloadWrapper m1;
  EXPECT_CALL(*m0.mock, CopyWithoutData()).WillOnce(Return(nullptr));
  EXPECT_CALL(*m1.mock, CopyWithoutData()).WillOnce(Return(nullptr));
  EXPECT_CALL(*m0.mock, GetData()).WillRepeatedly(Return(val0));
  EXPECT_CALL(*m1.mock, GetData()).WillRepeatedly(Return(val1));
  auto mpw = TestPayloadWrapper(m0, m1);

  auto payloadsNoData = mpw.CopyWithoutData();
  TestPayloadWrapper noDataCopy(payloadsNoData);
  const auto &noDataP0 = std::get<0>(noDataCopy.GetPayloads());
  const auto &noDataP1 = std::get<1>(noDataCopy.GetPayloads());
  const auto &p0 = std::get<0>(mpw.GetPayloads());
  const auto &p1 = std::get<1>(mpw.GetPayloads());

  ASSERT_EQ(noDataP0.mock.get(), nullptr);
  ASSERT_EQ(noDataP1.mock.get(), nullptr);

  ASSERT_NE(p0.mock.get(), nullptr);
  ASSERT_NE(p1.mock.get(), nullptr);
  ASSERT_EQ(p0.mock->GetData(), val0);
  ASSERT_EQ(p1.mock->GetData(), val1);
}

// TEST(TestPayloadWrapperTest, Subset_modifies_all_payloads)
// {
//   const int val0 = 10;
//   const int val1 = 42;
//   MockPayloadWrapper m0;
//   MockPayloadWrapper m1;
//   // This causes some error with the destruction of mocks, but tests work as expected
//   EXPECT_CALL(*m0.mock, Subset).WillOnce(Return(m0.mock));
//   EXPECT_CALL(*m1.mock, Subset).WillOnce(Return(m1.mock));
//   EXPECT_CALL(*m0.mock, GetData()).WillRepeatedly(Return(val0));
//   EXPECT_CALL(*m1.mock, GetData()).WillRepeatedly(Return(val1));
//   auto mpw = TestPayloadWrapper(std::move(m0), std::move(m1));
//
//   const size_t startN = 10;
//   const size_t count = 5;
//   TestPayloadWrapper subsetCopy = mpw.Subset(startN, count);
//
//   const auto &p0 = std::get<0>(mpw.GetPayloads());
//   const auto &p1 = std::get<1>(mpw.GetPayloads());
//   const auto &subsetP0 = std::get<0>(subsetCopy.GetPayloads());
//   const auto &subsetP1 = std::get<1>(subsetCopy.GetPayloads());
//
//   // Check that we created new instance
//   ASSERT_NE(p0.testId, subsetP0.testId);
//   ASSERT_NE(p1.testId, subsetP1.testId);
//
//   ASSERT_EQ(p0.GetData(), subsetP0.GetData());
//   ASSERT_EQ(p1.GetData(), subsetP1.GetData());
//
//   ASSERT_NE(p0.GetData(), p1.GetData());
//   ASSERT_NE(subsetP0.GetData(), subsetP1.GetData());
// }
