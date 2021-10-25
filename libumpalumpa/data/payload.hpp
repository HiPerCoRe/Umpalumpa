#pragma once

#include <stdexcept>
#include <ostream>
#include <iomanip>
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
      : ptr(d), info(ld), dataInfo(pd), description(desc)
    {}

    explicit Payload(const T &ld, const std::string &desc)
      : ptr(nullptr), info(ld), dataInfo(), description(desc + suffixEmpty)
    {}

    bool IsValid() const
    {
      // info and dataInfo must be always valid
      bool result = info.IsValid() && dataInfo.IsValid();
      if (IsEmpty()) return result;
      // it has some data, check the size
      return (result && HasValidBytes());
    }

    bool IsEmpty() const { return (nullptr == ptr) && (dataInfo.IsEmpty()); }

    // FIXME  it might be useful to have subset which takes e.g. a vector of possitions that we want
    // to get
    Payload Subset(size_t startN, size_t count) const// FIXME refactor
    {
      assert(!IsEmpty());
      size_t safeCount = 0;// change name to something more reasonable
      const auto newInfo = info.Subset(safeCount, startN, count);
      const auto offset = info.Offset(0, 0, 0, startN);

      const auto newDataInfo =
        PhysicalDescriptor(newInfo.Elems() * Sizeof(dataInfo.type), dataInfo.type);
      void *newData = reinterpret_cast<char *>(ptr) + (offset * Sizeof(dataInfo.type));
      const auto suffix = " [" + std::to_string(startN) + ".." + std::to_string(startN + safeCount);
      return Payload(newData, newInfo, newDataInfo, description + suffix);
    };

    /**
     * Create an exact copy of this Payload, but without data.
     * This can be useful for e.g. Algorithm initialization or to compare
     * multiple Payloads
     * */
    Payload CopyWithoutData() const
    {
      return Payload(
        nullptr, info, PhysicalDescriptor(0, dataInfo.type), description + suffixEmpty);
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
    bool IsEquivalentTo(const Payload<T> ref) const
    {
      return info.IsEquivalentTo(ref.info) && (dataInfo.type == ref.dataInfo.type);
    }

    /**
     * Returns minimal number of bytes necessary to fit this data.
     * Returned amount might be smaller than bytes provided by Physical descriptor,
     * as data represented by this Payload might not span the entire memory block.
     **/
    size_t GetBytes() { return info.GetPaddedSize().total * Sizeof(dataInfo.type); }

    // these shouold be private + getters / setters
    void *ptr;// constant pointer to non-constant data, type defined by other descriptors
    T info;
    PhysicalDescriptor dataInfo;
    std::string description;
    typedef T LDType;

  private:
    static auto constexpr suffixEmpty = " [empty]";
    bool HasValidBytes() const// FIXME refactor
    {
      return ((nullptr == ptr) && (0 == dataInfo.bytes))
             || (dataInfo.bytes >= (info.GetPaddedSize().total * Sizeof(dataInfo.type)));
    }

    template<typename DT>
    void
      PrivatePrint(std::ostream &out, const Size &total, const Size &dims, const Size &offset) const
    {

      auto original = out.flags();
      // prepare output formatting
      out << std::setfill(' ') << std::left << std::setprecision(3) << std::showpos;

      auto *data = reinterpret_cast<DT *>(ptr);
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
}// namespace data
}// namespace umpalumpa

template<typename T>
std::ostream &operator<<(std::ostream &out, const umpalumpa::data::Payload<T> &p)
{
  p.PrintData(out);
  return out;
}
