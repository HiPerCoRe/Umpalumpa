#pragma once

#include <libumpalumpa/data/logical_desriptor.hpp>
#include <libumpalumpa/data/physical_desriptor.hpp>

namespace umpalumpa {
namespace data {
  class Payload
  {
  public:
    Payload(void *d, const LogicalDescriptor &ld, const PhysicalDescriptor &pd) : data(d), info(ld), dataInfo(pd) {}
    void *const data;// constant pointer to non-constant data, type defined by physical and logical descriptor
    const LogicalDescriptor info;
    const PhysicalDescriptor dataInfo;
    bool IsValid() const { return info.IsValid() && IsValidBytes(); }

  private:
    bool IsValidBytes() const
    {
      return ((nullptr == data) && (0 == dataInfo.bytes))
             || (dataInfo.bytes >= (info.paddedSize.total * Sizeof(dataInfo.type)));
    }
  };
}// namespace data
}// namespace umpalumpa
