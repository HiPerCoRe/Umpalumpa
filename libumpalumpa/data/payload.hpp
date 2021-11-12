#pragma once

#include <stdexcept>
#include <ostream>
#include <iomanip>
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

  // Data need to be accessible from CPU
  // FIXME printing methods should be defined elsewhere
  void PrintData(std::ostream &out) const
  {
    Size offset(0, 0, 0, 0);
    auto dims = info.GetSize();
    PrintData(out, dims, offset);
  }

  // Data need to be accessible from CPU
  void PrintData(std::ostream &out, const Size &dims, const Size &offset) const
  {
    auto total = info.GetSize();
    switch (dataInfo.type) {
    case DataType::kFloat:
      PrivatePrint<float>(out, total, dims, offset);
      break;
    case DataType::kDouble:
      PrivatePrint<double>(out, total, dims, offset);
      break;
    default:
      throw std::logic_error("Trying to print unprintable type.");
    }
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

  /**
   * Returns minimal number of bytes necessary to fit this data.
   * Returned amount might be smaller than bytes provided by Physical descriptor,
   * as data represented by this Payload might not span the entire memory block.
   **/
  size_t GetRequiredBytes() const { return info.Elems() * Sizeof(dataInfo.GetType()); }

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

  template<typename DT>
  void
    PrivatePrint(std::ostream &out, const Size &total, const Size &dims, const Size &offset) const
  {

    auto original = out.flags();
    // prepare output formatting
    out << std::setfill(' ') << std::left << std::setprecision(3) << std::showpos;

    auto *data = reinterpret_cast<DT *>(GetPtr());
    for (size_t n = offset.n; n < offset.n + dims.n; n++) {
      for (size_t z = offset.z; z < offset.z + dims.z; z++) {
        for (size_t y = offset.y; y < offset.y + dims.y; y++) {
          for (size_t x = offset.x; x < offset.x + dims.x; x++) {
            auto index = n * total.single + z * total.y * total.x + y * total.x + x;
            out << std::setw(7) << data[index] << ' ';
          }
          out << '\n';
        }
        if (dims.z > 1 && z < dims.z - 1) out << "---\n";
      }
      if (dims.n > 1 && n < dims.n - 1) out << "###\n";
    }

    out.flags(original);
  }
};
}// namespace umpalumpa::data

template<typename T>
std::ostream &operator<<(std::ostream &out, const umpalumpa::data::Payload<T> &p)
{
  p.PrintData(out);
  return out;
}
