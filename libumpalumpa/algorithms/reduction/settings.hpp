#pragma once

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

private:
  Operation op = Operation::kPiecewiseSum;
};
}// namespace umpalumpa::reduction