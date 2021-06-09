#pragma once

namespace umpalumpa {
namespace data {

  enum class DataType {
    kVoid,
    kFloat,
    kDouble,
  };

  static size_t Sizeof(DataType t)
  {
    switch (t) {
    case DataType::kVoid:
      return 0;
    case DataType::kFloat:
      return sizeof(float);
    case DataType::kDouble:
      return sizeof(double);
    default:
      return 0;// unknown type
    }
  }
}// namespace data
}// namespace umpalumpa