/**
 * @file about_compilation.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2022-04-25
 *
 * MIT License
 *
 */

// More info: https://sourceforge.net/p/predef/wiki/Compilers/

#ifndef _COMPILATION_HPP_
#define _COMPILATION_HPP_

#include <string>

namespace my
{
namespace compile
{

const std::string arch();

const std::string compiler();

const std::string compiler_ver();

const std::string os();

const std::string cxx();

const std::string build_date();

const std::string arduino();

}  // namespace compile
}  // namespace my
#endif
