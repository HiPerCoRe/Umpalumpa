#pragma once

#include <string>
#include <libumpalumpa/data/physical_desriptor.hpp>

namespace umpalumpa::data {
/**
 * Basic storage unit of this framework.
 * It describes what and where are the data.
 * We do not manage your memory - no (de)allocation is directly done by this framework
 **/
template<typename T> class Payload
{
public:
  /**
   * Create a new Payload
   * ld describes content of the stored data
   * pd describes form of the stored data and their location
   * desc describes the data for debugging and tracking purposes
   **/
  explicit Payload(const T &ld, PhysicalDescriptor &&pd, const std::string &)
    : info(ld), dataInfo(std::move(pd))//, description(desc)
  {}

  /**
   * FIXME remove this constructor once we solve description transfer via StarPU
   **/
  explicit Payload(const T &ld, PhysicalDescriptor &&pd) : info(ld), dataInfo(std::move(pd)) {}

  bool IsValid() const
  {
    // info and dataInfo must be always valid
    bool result = info.IsValid() && dataInfo.IsValid();
    if (IsEmpty()) return result;
    // it has some data, check the size
    return result && (dataInfo.GetBytes() >= this->GetRequiredBytes());
  }

  bool IsEmpty() const { return dataInfo.IsEmpty(); }

  /**
   * Create an exact copy of this Payload, but without data.
   * This can be useful for e.g. Algorithm initialization or to compare
   * multiple Payloads
   * */
  Payload CopyWithoutData() const
  {
    return Payload(info, dataInfo.CopyWithoutData(),
      "");//, description + suffixEmpty);
  }

  /**
   * Returns true if this Payload is equivalent to reference one,
   * i.e. it has:
   * - equivalent Logical Descriptor
   *   - the size of this Paylod is the same, except for N,
   *     which can be lower or equal to reference
   * - the data type
   **/
  bool IsEquivalentTo(const Payload<T> &ref) const
  {
    return info.IsEquivalentTo(ref.info) && (dataInfo.GetType() == ref.dataInfo.GetType());
  }

  bool operator==(const Payload<T> &o) const { return info == o.info && dataInfo == o.dataInfo; }

  /**
   * Returns minimal number of bytes necessary to fit this data.
   * Returned amount might be smaller than bytes provided by Physical descriptor,
   * as data represented by this Payload might not span the entire memory block.
   **/
  size_t GetRequiredBytes() const { return info.Elems() * dataInfo.GetType().GetSize(); }

  inline void *GetPtr() const { return dataInfo.GetPtr(); }

  // these shouold be private + getters / setters

  T info;
  PhysicalDescriptor dataInfo;

  // FIXME we want to have Payload description, but it has
  // to be correclty handled by StarPU. StarPU thinks it's a pointer
  // so it has to be allocated / copied / freed separately to avoid invalid memory
  // operations
  // std::string description;

  typedef T LDType;

  /**
   * Use this with utmost causion and only when you have a very good reason,
   * e.g. you get existing Payload and you cannot change it.
   * Otherwise prefer to create a new Payload.
   **/
  void Set(const PhysicalDescriptor pd) { this->dataInfo = pd; }

private:
  static auto constexpr suffixEmpty = " [empty]";
};
}// namespace umpalumpa::data