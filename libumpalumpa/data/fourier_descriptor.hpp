#pragma once

#include <libumpalumpa/data/logical_desriptor.hpp>
#include <libumpalumpa/data/size.hpp>
#include <cassert>

namespace umpalumpa {
namespace data {
  class FourierDescriptor
  {
  public:
  // fixme say that padded size is size + padding
    explicit FourierDescriptor(const Size &s, const Size &padded)
      : size(s), paddedSize(padded), frequencyDomainSize(size.x / 2 + 1, size.y, size.z, size.n), frequencyDomainSizePadded(frequencyDomainSize)// TODO: need to somehow add description of how the data are padded
    {}
    virtual ~FourierDescriptor() {}
    bool IsValid() const { return paddedSize >= size; }
    virtual FourierDescriptor Subset(size_t &safeCount, const size_t startN, const size_t count) const
    {
      assert(this->IsValid());
      assert(startN <= paddedSize.n);
      safeCount = std::min(paddedSize.n - startN, count);
      return FourierDescriptor(size.CopyFor(safeCount), paddedSize.CopyFor(safeCount));
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

    // fixme these should be private + getters / setters
    Size size;
    Size paddedSize;
    Size frequencyDomainSize;
    Size frequencyDomainSizePadded;
    bool isSpatial; // FIXME: should be enum (can use the direction.hpp)
    bool isCentered; // FIXME: change to enum
    bool isNormalized; // FIXME: change to enum
    bool hasSymetry; // FIXME: find proper name
  };
}// namespace data
}// namespace umpalumpa
