#pragma once

#include <iostream>

namespace umpalumpa::reduction {

class Settings
{
public:
  enum class Operation {
    kPiecewiseSum// sum items from source and destination at the respective position and store them
                 // to destination
  };

  void SetOperation(const Operation &o) { op = o; }

  auto GetOperation() const { return op; }

  bool IsEquivalentTo(const Settings &ref) const { return op == ref.op; }

  void Serialize(std::ostream &out) const { out << static_cast<int>(op) << '\n'; }

  static auto Deserialize(std::istream &in)
  {
    int op;
    in >> op;
    Settings s;
    s.SetOperation(static_cast<Operation>(op));
    return s;
  }

private:
  Operation op = Operation::kPiecewiseSum;
};
}// namespace umpalumpa::reduction
