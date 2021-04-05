#pragma once

#include <libumpalumpa/data/size.hpp>
#include <cassert>
#include <string>

namespace umpalumpa {
namespace data {
  class LogicalDescriptor
  {
  public:
    explicit LogicalDescriptor(const Size &s, const Size &padded, const std::string &desc)
      : size(s), paddedSize(padded), description(desc)
    {}
    virtual ~LogicalDescriptor() {}
    bool IsValid() const { return paddedSize >= size; }
    virtual LogicalDescriptor Subset(const size_t startN, const size_t count) const
    {
      assert(this->IsValid());
      assert(startN <= paddedSize.n);
      const size_t safeCount = std::min(paddedSize.n - startN, count);
      return LogicalDescriptor(size.CopyFor(safeCount), paddedSize.CopyFor(safeCount), description);
    }

    virtual size_t Offset(size_t x, size_t y, size_t z, size_t n) const
    {
      assert(this->IsValid());
      assert(size.x >= x);
      assert(size.y >= y);
      assert(size.z >= z);
      assert(size.n >= n);
      return (n * paddedSize.single + z * (paddedSize.x * paddedSize.y) + y * (paddedSize.x) + x);
    }

    virtual size_t Elems() const { return paddedSize.total; }

    const Size size;
    const Size paddedSize;
    const std::string description;
  };
}// namespace data
}// namespace umpalumpa
