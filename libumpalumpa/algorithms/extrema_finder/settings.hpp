#pragma once

#include <libumpalumpa/algorithms/extrema_finder/location.hpp>
#include <libumpalumpa/algorithms/extrema_finder/result.hpp>
#include <libumpalumpa/algorithms/extrema_finder/extrema_type.hpp>
#include <libumpalumpa/algorithms/extrema_finder/precision.hpp>
#include <libumpalumpa/data/size.hpp>

namespace umpalumpa::extrema_finder {

class Settings
{
public:
  explicit Settings(const ExtremaType &t,
    const Location &l,
    const Result &r,
    const Precision &p = Precision::kSingle)
    : type(t), location(l), result(r), precision(p)
  {}

  auto GetType() const { return type; }

  auto GetLocation() const { return location; }

  auto GetResult() const { return result; }

  auto GetPrecision() const { return precision; }

  int GetVersion() const { return version; }

private:
  ExtremaType type;
  Location location;
  Result result;
  Precision precision;
  static constexpr int version =
    1;// FIXME add documentation that this must increase if we change settings
};

}// namespace umpalumpa::extrema_finder
