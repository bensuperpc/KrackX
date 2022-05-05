#include <string>

#include "gta_sa.hpp"
#ifdef BUILD_WITH_CUDA
#  include "cuda/kernel.hpp"
#endif
#include "gtest/gtest.h"

TEST(jamcrc, basic1)
{
  std::string str = "";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0xffffffff, GTA_SA::jamcrc(str));

#ifdef BUILD_WITH_CUDA
  EXPECT_NE(0x0, my::cuda::jamcrc(str.data(), str.size(), 0));
  EXPECT_EQ(0xffffffff, my::cuda::jamcrc(str.data(), str.size(), 0));
#endif
}

TEST(jamcrc, basic2)
{
  std::string str = "ASNAEB";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("BEANSA", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0x555fc201, GTA_SA::jamcrc(str));
#ifdef BUILD_WITH_CUDA
  EXPECT_NE(0x0, my::cuda::jamcrc(str.data(), str.size(), 0));
  EXPECT_EQ(0x555fc201, my::cuda::jamcrc(str.data(), str.size(), 0));
#endif
}

TEST(jamcrc, basic3)
{
  std::string str = "ASBHGRB";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("BRGHBSA", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0xa7613f99, GTA_SA::jamcrc(str));

#ifdef BUILD_WITH_CUDA
  EXPECT_NE(0x0, my::cuda::jamcrc(str.data(), str.size(), 0));
  EXPECT_EQ(0xa7613f99, my::cuda::jamcrc(str.data(), str.size(), 0));
#endif
}

TEST(jamcrc, basic4)
{
  std::string str = "XICWMD";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("DMWCIX", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0x1a9aa3d6, GTA_SA::jamcrc(str));

#ifdef BUILD_WITH_CUDA
  EXPECT_NE(0x0, my::cuda::jamcrc(str.data(), str.size(), 0));
  EXPECT_EQ(0x1a9aa3d6, my::cuda::jamcrc(str.data(), str.size(), 0));
#endif
}

TEST(jamcrc, basic5)
{
  std::string str = "LGBTQIA+";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("+AIQTBGL", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0x6ba88a6, GTA_SA::jamcrc(str));

#ifdef BUILD_WITH_CUDA
  EXPECT_NE(0x0, my::cuda::jamcrc(str.data(), str.size(), 0));
  EXPECT_EQ(0x6ba88a6, my::cuda::jamcrc(str.data(), str.size(), 0));
#endif
}

TEST(jamcrc, basic6)
{
  std::string str = "intergouvernementalisations";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("snoitasilatnemenrevuogretni", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0x1a384955, GTA_SA::jamcrc(str));

#ifdef BUILD_WITH_CUDA
  EXPECT_NE(0x0, my::cuda::jamcrc(str.data(), str.size(), 0));
  EXPECT_EQ(0x1a384955, my::cuda::jamcrc(str.data(), str.size(), 0));
#endif
}
