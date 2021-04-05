#pragma once

#include <libumpalumpa/data/logical_desriptor.hpp>
#include <libumpalumpa/data/physical_desriptor.hpp>
#include <cassert>
#include <type_traits>

namespace umpalumpa {
namespace data {
  template<typename T>
  class Payload
  {

    static_assert(std::is_base_of<LogicalDescriptor, T>::value, "T must inherit from Logical Descriptor");
  public:
    explicit Payload(void *d, const T &ld, const PhysicalDescriptor &pd)
      : data(d), info(ld), dataInfo(pd)
    {}
    virtual ~Payload() = default;
    void *const data;// constant pointer to non-constant data, type defined by physical and logical descriptor
    const T info;
    const PhysicalDescriptor dataInfo;
    bool IsValid() const { return info.IsValid() && IsValidBytes(); }
    Payload Subset(size_t startN, size_t count) const
    {
      assert(this->IsValid());
      const auto newInfo = info.Subset(startN, count);
      const auto offset = info.Offset(0, 0, 0, startN);

      const auto newDataInfo = PhysicalDescriptor(newInfo.Elems() * Sizeof(dataInfo.type), dataInfo.type);
      void *newData = reinterpret_cast<char *>(data) + (offset * Sizeof(dataInfo.type));
      return Payload(newData, newInfo, newDataInfo);
    };

  private:
    bool IsValidBytes() const
    {
      return ((nullptr == data) && (0 == dataInfo.bytes))
             || (dataInfo.bytes >= (info.paddedSize.total * Sizeof(dataInfo.type)));
    }
  };
}// namespace data
}// namespace umpalumpa
