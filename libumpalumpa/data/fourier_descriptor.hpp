#pragma once

#include <libumpalumpa/data/logical_desriptor.hpp>
#include <libumpalumpa/data/size.hpp>
#include <cassert>
#include <memory>
#include <optional>
#include <exception>

namespace umpalumpa {
namespace data {
  class FourierDescriptor
  {
  public:
    struct FourierSpaceDescriptor
    {
      bool isCentered;// FIXME: change to enum
      bool isNormalized;// FIXME: change to enum
      bool hasSymetry;// FIXME: find proper name
      bool operator==(const FourierSpaceDescriptor &o) const
      {
        return (isCentered == o.isCentered) && (isNormalized == o.isNormalized)
               && (hasSymetry == o.hasSymetry);
      }
    };

    inline const auto &GetSpatialSize() const { return size; }

    inline const auto &GetPaddedSpatialSize() const { return paddedSize; }

    inline const auto &GetFrequencySize() const { return frequencyDomainSize; }

    inline const auto &GetPaddedFrequencySize() const { return frequencyDomainSizePadded; }

    inline const auto &GetSize() const
    {
      if (isSpatial) { return size; }
      return frequencyDomainSize;
    }

    inline const auto &GetPaddedSize() const
    {
      if (isSpatial) { return paddedSize; }
      return frequencyDomainSizePadded;
    }

    const auto &GetFourierSpaceDescriptor() const
    {
      return fsd;// FIXME decide whether you want to throw exception, or return optional
    }

    double GetNormFactor() const { return 1.0 / static_cast<double>(paddedSize.single); }

    // fixme say that padded size is size + padding
    explicit FourierDescriptor(const Size &s, const Size &padded)
      : size(s), paddedSize(padded), frequencyDomainSize(ComputeFrequencySize(s)),
        frequencyDomainSizePadded(frequencyDomainSize),
        isSpatial(true)// TODO: need to somehow add description of how the data are padded
    {}
    explicit FourierDescriptor(const Size &s, const Size &padded, const FourierSpaceDescriptor &d)
      : size(s), paddedSize(padded), frequencyDomainSize(ComputeFrequencySize(s)),
        frequencyDomainSizePadded(frequencyDomainSize), isSpatial(false),
        fsd(d)// TODO: need to somehow add description of how the data are padded
    {}
    virtual ~FourierDescriptor() {}
    bool IsValid() const { return paddedSize >= size; }
    virtual FourierDescriptor
      Subset(size_t &safeCount, const size_t startN, const size_t count) const
    {
      assert(this->IsValid());
      assert(startN <= GetPaddedSize().n);
      safeCount = std::min(GetPaddedSize().n - startN, count);
      if (fsd) {
        return FourierDescriptor(
          size.CopyFor(safeCount), paddedSize.CopyFor(safeCount), fsd.value());
      }
      return FourierDescriptor(size.CopyFor(safeCount), paddedSize.CopyFor(safeCount));
    }

    virtual size_t Offset(size_t x, size_t y, size_t z, size_t n) const
    {
      assert(this->IsValid());
      assert(GetSize().x >= x);
      assert(GetSize().y >= y);
      assert(GetSize().z >= z);
      assert(GetSize().n >= n);
      return (n * GetPaddedSize().single + z * (GetPaddedSize().x * GetPaddedSize().y)
              + y * (GetPaddedSize().x) + x);
    }

    virtual size_t Elems() const { return GetPaddedSize().total; }

    bool IsEquivalentTo(const FourierDescriptor &ref) const
    {
      return size.IsEquivalentTo(ref.size) && paddedSize.IsEquivalentTo(ref.paddedSize)
             && (isSpatial == ref.isSpatial) && (fsd == ref.fsd);
    }

    // fixme these should be private + getters / setters
  private:
    Size size;
    Size paddedSize;
    Size frequencyDomainSize;
    Size frequencyDomainSizePadded;
    bool isSpatial;// FIXME: should be enum (can use the direction.hpp)
    std::optional<FourierSpaceDescriptor> fsd;

    Size ComputeFrequencySize(const Size &s) { return Size(s.x / 2 + 1, s.y, s.z, s.n); }
  };
}// namespace data
}// namespace umpalumpa
