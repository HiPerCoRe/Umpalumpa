#pragma once

#include <string>
#include <libumpalumpa/data/physical_desriptor.hpp>
#include <cassert>

namespace umpalumpa {
namespace data {
  template<typename T> class Payload
  {
  public:
  // add documentation, especially that we don't manage data
    explicit Payload(void *d, const T &ld, const PhysicalDescriptor &pd, const std::string &desc)
      : data(d), info(ld), dataInfo(pd), description(desc)
    {}

    explicit Payload(const T &ld, const std::string &desc)
      : data(nullptr), info(ld), dataInfo(), description(desc + suffixEmpty)
    {}

    bool IsValid() const
    {
      // info and dataInfo must be always valid
      bool result = info.IsValid() && dataInfo.IsValid();
      if (IsEmpty()) return result;
      // it has some data, check the size
      return (result && HasValidBytes());
    }

    bool IsEmpty() const { return (nullptr == data) && (dataInfo.IsEmpty()); }

  // FIXME  it might be useful to have subset which takes e.g. a vector of possitions that we want to get
    Payload Subset(size_t startN, size_t count) const// FIXME refactor
    {
      assert(!IsEmpty());
      size_t safeCount = 0; // change name to something more reasonable
      const auto newInfo = info.Subset(safeCount, startN, count);
      const auto offset = info.Offset(0, 0, 0, startN);

      const auto newDataInfo =
        PhysicalDescriptor(newInfo.Elems() * Sizeof(dataInfo.type), dataInfo.type);
      void *newData = reinterpret_cast<char *>(data) + (offset * Sizeof(dataInfo.type));
      const auto suffix = " [" + std::to_string(startN) + ".." + std::to_string(startN + safeCount);
      return Payload(newData, newInfo, newDataInfo, description + suffix);
    };

    Payload CopyWithoutData() const
    {
      return Payload(
        nullptr, info, PhysicalDescriptor(0, dataInfo.type), description + suffixEmpty);
    }

    // these shouold be private + getters / setters
    void *data;// constant pointer to non-constant data, type defined by other descriptors
    T info;
    PhysicalDescriptor dataInfo;
    std::string description;
    typedef T type;

  private:
    static auto constexpr suffixEmpty = " [empty]";
    bool HasValidBytes() const// FIXME refactor
    {
      return ((nullptr == data) && (0 == dataInfo.bytes))
             || (dataInfo.bytes >= (info.GetPaddedSize().total * Sizeof(dataInfo.type)));
    }
  };
}// namespace data
}// namespace umpalumpa
