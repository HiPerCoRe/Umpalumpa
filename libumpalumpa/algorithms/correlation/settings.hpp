#pragma once
#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/algorithms/correlation/correlation_type.hpp>

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

private:
  static constexpr int version = 1;
  CorrelationType type;
  bool center;
};
}// namespace umpalumpa::correlation
