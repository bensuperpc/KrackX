#include <array>
#include <string>

#include "gtest/gtest.h"

TEST(basic, basic1)
{
  EXPECT_NE("AA", "A");

  EXPECT_EQ("A", "A");
}