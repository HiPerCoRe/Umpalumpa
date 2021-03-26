#pragma once

#include <libumpalumpa/algorithms/extrema_finder/search_location.hpp>
#include <libumpalumpa/algorithms/extrema_finder/search_result.hpp>
#include <libumpalumpa/algorithms/extrema_finder/search_type.hpp>
#include <libumpalumpa/data/size.hpp>

namespace umpalumpa {
namespace extrema_finder {

  class Settings
  {
  public:
    explicit Settings(const SearchType &t, const SearchLocation &l, const SearchResult &r)
      : type(t), location(l), result(r)
    {}

    const SearchType type;
    const SearchLocation location;
    const SearchResult result;
    bool dryRun;
    static constexpr int version = 1;
  };

}// namespace extrema_finder
}// namespace umpalumpa