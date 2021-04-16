#pragma once

#include <string>

namespace umpalumpa {
namespace utils {
  std::string GetExecPath();

  constexpr char kPathSeparator =
#ifdef _WIN32
    '\\';
#else
    '/';
#endif

  std::string Canonize(const std::string &p);

  std::string Exec(const std::string &cmd);
}// namespace utils
}// namespace umpalumpa
