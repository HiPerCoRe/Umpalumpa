#pragma once

#include <libumpalumpa/algorithms/extrema_finder/search_location.hpp>
#include <libumpalumpa/algorithms/extrema_finder/search_result.hpp>
#include <libumpalumpa/algorithms/extrema_finder/search_type.hpp>
#include <libumpalumpa/data/size.hpp>

namespace umpalumpa::extrema_finder {

class Settings
{
public:
  explicit Settings(const SearchType &t, const SearchLocation &l, const SearchResult &r)
    : type(t), location(l), result(r)
  {}

  auto GetType() const { return type; }

  auto GetLocation() const { return location; }

  auto GetResult() const { return result; }

  int GetVersion() const { return version; }

private:
  SearchType type;
  SearchLocation location;
  SearchResult result;
  static constexpr int version =
    1;// FIXME add documentation that this must increase if we change settings
};

}// namespace umpalumpa::extrema_finder
