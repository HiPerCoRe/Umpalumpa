#pragma once

#include <iostream>

namespace umpalumpa::initialization {

class Settings
{
public:
  bool IsEquivalentTo(const Settings &) const { return true; }

  void Serialize(std::ostream &out) const { out << '\n'; }

  static auto Deserialize(std::istream &) { return Settings{}; }
};
}// namespace umpalumpa::initialization
