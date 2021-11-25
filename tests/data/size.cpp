#include <libumpalumpa/data/size.hpp>

#include <gtest/gtest.h>

using umpalumpa::data::Size;
using umpalumpa::data::Dimensionality;

TEST(SizeTests, Validity)
{
  EXPECT_FALSE(Size(0, 0, 0, 0).IsValid());
  EXPECT_FALSE(Size(0, 0, 0, 1).IsValid());
  EXPECT_FALSE(Size(0, 0, 1, 0).IsValid());
  EXPECT_FALSE(Size(0, 0, 1, 1).IsValid());

  EXPECT_FALSE(Size(0, 1, 0, 0).IsValid());
  EXPECT_FALSE(Size(0, 1, 0, 1).IsValid());
  EXPECT_FALSE(Size(0, 1, 1, 0).IsValid());
  EXPECT_FALSE(Size(0, 1, 1, 1).IsValid());

  EXPECT_FALSE(Size(1, 0, 0, 0).IsValid());
  EXPECT_FALSE(Size(1, 0, 0, 1).IsValid());
  EXPECT_FALSE(Size(1, 0, 1, 0).IsValid());
  EXPECT_FALSE(Size(1, 0, 1, 1).IsValid());

  EXPECT_FALSE(Size(1, 1, 0, 0).IsValid());
  EXPECT_FALSE(Size(1, 1, 0, 1).IsValid());
  EXPECT_FALSE(Size(1, 1, 1, 0).IsValid());
  EXPECT_TRUE(Size(1, 1, 1, 1).IsValid());

  EXPECT_FALSE(Size(0, 0, 0, 0).IsValid());
  EXPECT_FALSE(Size(0, 0, 0, 2).IsValid());
  EXPECT_FALSE(Size(0, 0, 2, 0).IsValid());
  EXPECT_FALSE(Size(0, 0, 2, 2).IsValid());

  EXPECT_FALSE(Size(0, 2, 0, 0).IsValid());
  EXPECT_FALSE(Size(0, 2, 0, 2).IsValid());
  EXPECT_FALSE(Size(0, 2, 2, 0).IsValid());
  EXPECT_FALSE(Size(0, 2, 2, 2).IsValid());

  EXPECT_FALSE(Size(2, 0, 0, 0).IsValid());
  EXPECT_FALSE(Size(2, 0, 0, 2).IsValid());
  EXPECT_FALSE(Size(2, 0, 2, 0).IsValid());
  EXPECT_FALSE(Size(2, 0, 2, 2).IsValid());

  EXPECT_FALSE(Size(2, 2, 0, 0).IsValid());
  EXPECT_FALSE(Size(2, 2, 0, 2).IsValid());
  EXPECT_FALSE(Size(2, 2, 2, 0).IsValid());
  EXPECT_TRUE(Size(2, 2, 2, 2).IsValid());

  EXPECT_TRUE(Size(1, 1, 1, 1).IsValid());
  EXPECT_TRUE(Size(1, 1, 1, 2).IsValid());
  EXPECT_FALSE(Size(1, 1, 2, 1).IsValid());
  EXPECT_FALSE(Size(1, 1, 2, 2).IsValid());

  EXPECT_FALSE(Size(1, 2, 1, 1).IsValid());
  EXPECT_FALSE(Size(1, 2, 1, 2).IsValid());
  EXPECT_FALSE(Size(1, 2, 2, 1).IsValid());
  EXPECT_FALSE(Size(1, 2, 2, 2).IsValid());

  EXPECT_TRUE(Size(2, 1, 1, 1).IsValid());
  EXPECT_TRUE(Size(2, 1, 1, 2).IsValid());
  EXPECT_FALSE(Size(2, 1, 2, 1).IsValid());
  EXPECT_FALSE(Size(2, 1, 2, 2).IsValid());

  EXPECT_TRUE(Size(2, 2, 1, 1).IsValid());
  EXPECT_TRUE(Size(2, 2, 1, 2).IsValid());
  EXPECT_TRUE(Size(2, 2, 2, 1).IsValid());
  EXPECT_TRUE(Size(2, 2, 2, 2).IsValid());
}

TEST(SizeTests, Dimensionality)
{
  EXPECT_EQ(Size(0, 1, 1, 1).GetDim(), Dimensionality::kInvalid);
  EXPECT_EQ(Size(1, 1, 1, 1).GetDim(), Dimensionality::k1Dim);
  EXPECT_EQ(Size(2, 1, 1, 1).GetDim(), Dimensionality::k1Dim);

  EXPECT_EQ(Size(0, 1, 1, 1).GetDim(), Dimensionality::kInvalid);
  EXPECT_EQ(Size(1, 2, 1, 1).GetDim(), Dimensionality::kInvalid);
  EXPECT_EQ(Size(2, 2, 1, 1).GetDim(), Dimensionality::k2Dim);
  EXPECT_EQ(Size(3, 2, 1, 1).GetDim(), Dimensionality::k2Dim);

  EXPECT_EQ(Size(0, 1, 0, 1).GetDim(), Dimensionality::kInvalid);
  EXPECT_EQ(Size(1, 2, 1, 1).GetDim(), Dimensionality::kInvalid);
  EXPECT_EQ(Size(2, 2, 2, 1).GetDim(), Dimensionality::k3Dim);
  EXPECT_EQ(Size(3, 2, 3, 1).GetDim(), Dimensionality::k3Dim);
}

TEST(SizeTests, Single)
{
  EXPECT_EQ(Size(1, 1, 1, 1).single, 1);
  EXPECT_EQ(Size(2, 1, 1, 1).single, 2);

  EXPECT_EQ(Size(2, 2, 1, 1).single, 4);
  EXPECT_EQ(Size(3, 2, 1, 1).single, 6);

  EXPECT_EQ(Size(2, 2, 2, 1).single, 8);
  EXPECT_EQ(Size(3, 2, 3, 1).single, 18);

  EXPECT_EQ(Size(1, 1, 1, 2).single, 1);
  EXPECT_EQ(Size(2, 1, 1, 3).single, 2);

  EXPECT_EQ(Size(2, 2, 1, 4).single, 4);
  EXPECT_EQ(Size(3, 2, 1, 5).single, 6);

  EXPECT_EQ(Size(2, 2, 2, 6).single, 8);
  EXPECT_EQ(Size(3, 2, 3, 7).single, 18);
}

TEST(SizeTests, Total)
{
  EXPECT_EQ(Size(1, 1, 1, 1).total, 1);
  EXPECT_EQ(Size(2, 1, 1, 1).total, 2);
  EXPECT_EQ(Size(1, 1, 1, 5).total, 5);
  EXPECT_EQ(Size(2, 1, 1, 5).total, 10);

  EXPECT_EQ(Size(2, 2, 1, 1).total, 4);
  EXPECT_EQ(Size(3, 2, 1, 1).total, 6);
  EXPECT_EQ(Size(2, 2, 1, 3).total, 12);
  EXPECT_EQ(Size(3, 2, 1, 3).total, 18);

  EXPECT_EQ(Size(2, 2, 2, 1).total, 8);
  EXPECT_EQ(Size(3, 2, 3, 1).total, 18);
  EXPECT_EQ(Size(2, 2, 2, 7).total, 56);
  EXPECT_EQ(Size(3, 2, 3, 7).total, 126);
}
