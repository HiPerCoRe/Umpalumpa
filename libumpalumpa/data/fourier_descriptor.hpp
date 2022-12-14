#pragma once

#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/data/padding_descriptor.hpp>
#include <cassert>
#include <memory>
#include <optional>
#include <exception>
#include <iostream>

namespace umpalumpa {
namespace data {
  class FourierDescriptor
  {
  public:
    struct FourierSpaceDescriptor
    {
      bool isCentered;// data has been shifted in the spatial domain, i.e. phase has been altered //
                      // FIXME: change to enum
      bool isShifted;// data has been shifted in the fourier domain, position of the coefficient has
                     // been altered // FIXME: change to enum
      bool isNormalized;// FIXME: change to enum
      bool hasSymetry = false;// FIXME: find proper name // if true, the whole space is stored.
                              // Otherwise only the non-redundant half is stored
      bool operator==(const FourierSpaceDescriptor &o) const
      {
        return (isCentered == o.isCentered) && (isShifted == o.isShifted)
               && (isNormalized == o.isNormalized) && (hasSymetry == o.hasSymetry);
      }

      void Serialize(std::ostream &out) const
      {
        out << isCentered << ' ' << isShifted << ' ' << isNormalized << ' ' << hasSymetry << '\n';
      }
      static auto Deserialize(std::istream &in)
      {
        bool c, sh, n, sy;
        in >> c >> sh >> n >> sy;
        return FourierSpaceDescriptor{ c, sh, n, sy };
      }
    };

    /**
     * Constructor for data which are still in Spatial domain.
     * This constructor assumes no padding is present.
     * Size is the size in the Spatial domain, i.e. before transformation to Fourier space
     **/
    explicit FourierDescriptor(const Size &s)
      : size(s), paddedSize(s), frequencyDomainSize(ComputeFrequencySize(s, {})),
        frequencyDomainSizePadded(frequencyDomainSize), padding(PaddingDescriptor()),
        frequencyDomainPadding(padding), isSpatial(true)
    {}

    /**
     * Constructor for data which are still in Spatial domain.
     * Size is the size in the Spatial domain, i.e. before transformation to Fourier space
     **/
    explicit FourierDescriptor(const Size &s, const PaddingDescriptor &p)
      : size(s), paddedSize(ComputePaddedSize(s, p)),
        frequencyDomainSize(ComputeFrequencySize(s, {})),
        frequencyDomainSizePadded(frequencyDomainSize), padding(p), frequencyDomainPadding(p),
        isSpatial(true)
    {}

    /**
     * Constructor for data which are already converted to Fourier space.
     * Size is the size in the Spatial domain, i.e. before transformation to Fourier space
     **/
    explicit FourierDescriptor(const Size &s,
      const PaddingDescriptor &p,
      const FourierSpaceDescriptor &d)
      : size(s), paddedSize(ComputePaddedSize(s, p)),
        frequencyDomainSize(ComputeFrequencySize(s, d)),
        frequencyDomainSizePadded(frequencyDomainSize), padding(PaddingDescriptor()),
        frequencyDomainPadding(padding), isSpatial(false), fsd(d)
    {}

    virtual ~FourierDescriptor() {}

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

    inline bool IsPadded() const { return GetSize() != GetPaddedSize(); }

    inline const auto &GetPadding() const
    {
      if (isSpatial) { return padding; }
      return frequencyDomainPadding;
    }

    const auto &GetFourierSpaceDescriptor() const
    {
      return fsd;// FIXME decide whether you want to throw exception, or return optional
    }

    double GetNormFactor() const { return 1.0 / static_cast<double>(paddedSize.single); }

    bool IsValid() const { return size.IsValid() && paddedSize >= size; }

    virtual FourierDescriptor
      Subset(size_t &safeCount, const size_t startN, const size_t count) const
    {
      assert(this->IsValid());
      assert(startN <= GetPaddedSize().n);
      safeCount = std::min(GetPaddedSize().n - startN, count);
      if (fsd) { return FourierDescriptor(size.CopyFor(safeCount), padding, fsd.value()); }
      return FourierDescriptor(size.CopyFor(safeCount), padding);
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
      return size.IsEquivalentTo(ref.size) && (padding == ref.padding)
             && (frequencyDomainPadding == ref.frequencyDomainPadding)
             && (isSpatial == ref.isSpatial) && (fsd == ref.fsd);
    }

    bool operator==(const FourierDescriptor &o) const
    {
      return size == o.size && paddedSize == o.paddedSize
             && frequencyDomainSize == o.frequencyDomainSize
             && frequencyDomainSizePadded == o.frequencyDomainSizePadded && padding == o.padding
             && frequencyDomainSizePadded == o.frequencyDomainSizePadded && isSpatial == o.isSpatial
             && fsd == o.fsd;
    }

    void Serialize(std::ostream &out) const
    {
      size.Serialize(out);
      padding.Serialize(out);
      out << isSpatial << ' ';
      if (!isSpatial) {
        fsd.value().Serialize(out);
      } else {
        out << '\n';
      }
    }
    static auto Deserialize(std::istream &in)
    {
      bool isSpatial;
      auto size = Size::Deserialize(in);
      auto padding = PaddingDescriptor::Deserialize(in);
      in >> isSpatial;
      if (!isSpatial) {
        auto fsd = FourierSpaceDescriptor::Deserialize(in);
        return FourierDescriptor(size, padding, fsd);
      }
      return FourierDescriptor(size, padding);
    }

  private:
    Size ComputeFrequencySize(const Size &s, const FourierSpaceDescriptor &d)
    {
      return Size((d.hasSymetry ? s.x : s.x / 2 + 1), s.y, s.z, s.n);
    }

    Size ComputePaddedSize(const Size &s, const PaddingDescriptor &p) const
    {
      return Size(s.x + p.GetXBeg() + p.GetXEnd(),
        s.y + p.GetYBeg() + p.GetYEnd(),
        s.z + p.GetZBeg() + p.GetZEnd(),
        s.n);
    }

    Size size;
    Size paddedSize;
    Size frequencyDomainSize;
    Size frequencyDomainSizePadded;
    PaddingDescriptor padding;
    PaddingDescriptor frequencyDomainPadding;
    bool isSpatial;// FIXME: should be enum (can use the direction.hpp)
    std::optional<FourierSpaceDescriptor> fsd;
  };
}// namespace data
}// namespace umpalumpa
