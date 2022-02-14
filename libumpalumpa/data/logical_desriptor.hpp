#pragma once

#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/data/padding_descriptor.hpp>
#include <cassert>
#include <string>
#include <iostream>

namespace umpalumpa::data {
class LogicalDescriptor
{
public:
  /**
   * This constructor assumes no padding is present
   **/
  explicit LogicalDescriptor(const Size &s) : size(s), paddedSize(s), padding(PaddingDescriptor())
  {}

  explicit LogicalDescriptor(const Size &s, const PaddingDescriptor &p)
    : size(s), paddedSize(ComputePaddedSize(s, p)), padding(p)
  {}

  virtual ~LogicalDescriptor() {}

  bool IsValid() const { return size.IsValid() && paddedSize >= size; }

  virtual LogicalDescriptor Subset(size_t &safeCount, const size_t startN, const size_t count) const
  {
    assert(this->IsValid());
    assert(startN <= paddedSize.n);
    safeCount = std::min(paddedSize.n - startN, count);
    return LogicalDescriptor(size.CopyFor(safeCount), padding);
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

  inline const auto &GetPaddedSize() const { return paddedSize; }

  inline bool IsPadded() const { return size != paddedSize; }

  inline const auto &GetPadding() const { return padding; }

  inline const auto &GetSize() const { return size; }

  virtual size_t Elems() const { return paddedSize.total; }

  bool IsEquivalentTo(const LogicalDescriptor &ref) const
  {
    return size.IsEquivalentTo(ref.size) && (padding == ref.padding);
  }

  void Serialize(std::ostream &out) const
  {
    size.Serialize(out);
    padding.Serialize(out);
  }
  static auto Deserialize(std::istream &in)
  {
    // Needs to be written like this to guarantee the order of execution!
    // Do NOT try to refactor this into one-liner!!!
    auto size = Size::Deserialize(in);
    auto padding = PaddingDescriptor::Deserialize(in);
    return LogicalDescriptor(size, padding);
  }


  bool operator==(const LogicalDescriptor &o) const
  {
    return size == o.size && paddedSize == o.paddedSize && padding == o.padding;
  }

private:
  Size ComputePaddedSize(const Size &s, const PaddingDescriptor &p) const
  {
    return Size(s.x + p.GetXBeg() + p.GetXEnd(),
      s.y + p.GetYBeg() + p.GetYEnd(),
      s.z + p.GetZBeg() + p.GetZEnd(),
      s.n);
  }

  Size size;
  Size paddedSize;
  PaddingDescriptor padding;
};
}// namespace umpalumpa::data
