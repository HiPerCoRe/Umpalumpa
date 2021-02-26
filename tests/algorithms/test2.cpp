#include <libumpalumpa/algorithms/alg1.hpp>
#include <iostream>
#include "gtest/gtest.h"


TEST(example, myTest2)
{
  auto tmp = libumpalumpa::Umpalumpa();
  ASSERT_GT(tmp.speak(), 0);
}


	