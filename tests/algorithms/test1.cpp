#include <libumpalumpa/algorithms/alg1.hpp>
#include <iostream>
#include "gtest/gtest.h"


TEST(TestAlgs, myTest1)
{
  auto tmp = libumpalumpa::Umpalumpa();
  ASSERT_EQ(tmp.speak(), 1);
}


