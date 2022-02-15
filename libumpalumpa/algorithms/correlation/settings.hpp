#pragma once
#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/algorithms/correlation/correlation_type.hpp>
#include <iostream>

namespace umpalumpa::correlation {
class Settings
{
public:
  Settings(CorrelationType t) : type(t), center(true) {}

  bool IsEquivalentTo(const Settings &ref) const
  {
    return type == ref.type && center == ref.center;
  }

  int GetVersion() const { return version; }

  void SetCenter(bool val) { this->center = val; }
  bool GetCenter() const { return center; }

  void SetCorrelationType(CorrelationType val) { this->type = val; }
  CorrelationType GetCorrelationType() const { return type; }

  void Serialize(std::ostream &out) const
  {
    out << static_cast<int>(type) << ' ' << center << '\n';
  }
  static auto Deserialize(std::istream &in)
  {
    int type;
    bool center;
    in >> type >> center;
    auto s = Settings(static_cast<CorrelationType>(type));
    s.SetCenter(center);
    return s;
  }

private:
  static constexpr int version = 1;
  CorrelationType type;
  bool center;
};
}// namespace umpalumpa::correlation
