#pragma once

namespace umpalumpa {
namespace data {

  enum class DataType { kVoid, kFloat, kDouble, kComplexFloat, kComplexDouble };

  static size_t Sizeof(DataType t)
  {
    switch (t) {
    case DataType::kVoid:
      return 0;
    case DataType::kFloat:
      return sizeof(float);
    case DataType::kDouble:
      return sizeof(double);
    case DataType::kComplexFloat:
      return sizeof(float) * 2;
    case DataType::kComplexDouble:
      return sizeof(double) * 2;
    default:
      return 0;// unknown type
    }
  }
}// namespace data
}// namespace umpalumpa