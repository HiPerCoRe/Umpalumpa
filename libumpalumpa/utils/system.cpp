
#include <libumpalumpa/utils/system.hpp>
#include <libumpalumpa/utils/logger.hpp>
#include <unistd.h>
#include <linux/limits.h>

namespace umpalumpa {
namespace utils {
  std::string GetExecPath()
  {
    char result[PATH_MAX];
    auto count = static_cast<size_t>(readlink("/proc/self/exe", result, PATH_MAX));
    if (count > 0) return std::string(result, count) + kPathSeparator;
    spdlog::error("Could not get current executable path (via readlink /proc/self/exe)");
    return "";
  }

  std::string Exec(const std::string &cmd)
  {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) { throw std::runtime_error("popen() failed!"); }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) { result += buffer.data(); }
    return result;// includes EOL
  }

  std::string Canonize(const std::string &p) { return Exec("readlink -m " + p); }

  std::string GetSourceFilePath(const std::string &relPath)
  {
    std::string fullPath = utils::GetExecPath() + relPath;
    std::string canonicalPath = utils::Canonize(fullPath);
    canonicalPath.erase(
      std::remove(canonicalPath.begin(), canonicalPath.end(), '\n'), canonicalPath.end());
    return canonicalPath;
  }
}// namespace utils
}// namespace umpalumpa