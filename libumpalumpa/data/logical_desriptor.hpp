#pragma once

#include <libumpalumpa/data/size.hpp>

namespace umpalumpa {
namespace data {
  class LogicalDescriptor
  {
  public:
    LogicalDescriptor(const Size &s, const Size &padded) : size(s), paddedSize(padded) {}
    bool IsValid() const { return paddedSize >= size; }

    const Size size;
    const Size paddedSize;
  };
}// namespace data
}// namespace umpalumpa
