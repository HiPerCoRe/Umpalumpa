#pragma once

#include <libumpalumpa/operations/extrema_finder/location.hpp>
#include <libumpalumpa/operations/extrema_finder/result.hpp>
#include <libumpalumpa/operations/extrema_finder/extrema_type.hpp>
#include <libumpalumpa/operations/extrema_finder/precision.hpp>
#include <libumpalumpa/data/size.hpp>
#include <iostream>

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

  bool IsEquivalentTo(const Settings &ref) const
  {
    return type == ref.type && location == ref.location && result == ref.result
           && precision == ref.precision;
  }

  auto GetType() const { return type; }

  auto GetLocation() const { return location; }

  auto GetResult() const { return result; }

  auto GetPrecision() const { return precision; }

  int GetVersion() const { return version; }

  void Serialize(std::ostream &out) const
  {
    out << static_cast<int>(type) << ' ' << static_cast<int>(location) << ' '
        << static_cast<int>(result) << ' ' << static_cast<int>(precision) << '\n';
  }
  static auto Deserialize(std::istream &in)
  {
    int t, l, r, p;
    in >> t >> l >> r >> p;
    return Settings(static_cast<ExtremaType>(t),
      static_cast<Location>(l),
      static_cast<Result>(r),
      static_cast<Precision>(p));
  }

private:
  ExtremaType type;
  Location location;
  Result result;
  Precision precision;
  static constexpr int version =
    1;// FIXME add documentation that this must increase if we change settings
};

}// namespace umpalumpa::extrema_finder
