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

  // Path to repository (defined at compile time)
  inline const std::string kProjectRoot = PROJECT_ROOT;

  std::string Canonize(const std::string &p);

  std::string Exec(const std::string &cmd);

  std::string GetSourceFilePath(const std::string &relPath);
}// namespace utils
}// namespace umpalumpa
