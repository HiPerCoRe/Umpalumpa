#pragma once

#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/data/data_type.hpp>

namespace umpalumpa::data {
class PhysicalDescriptor
{
public:
  explicit PhysicalDescriptor(void *data, size_t b, DataType dataType)
    : ptr(data), bytes(b), type(dataType){};

  explicit PhysicalDescriptor() : PhysicalDescriptor(nullptr, 0, DataType::kVoid){};

  inline size_t GetBytes() const { return bytes; }

  inline float GetKBytes() const { return static_cast<float>(bytes) / 1024.f; }

  inline float GetMBytes() const { return static_cast<float>(bytes) / (1024.f * 1024.f); }

  inline float GetGBytes() const { return static_cast<float>(bytes) / (1024.f * 1024.f * 1024.f); }

  inline DataType GetType() const { return type; }

  inline void *GetPtr() const { return ptr; }

  /**
   * Descriptor is valid if it describes empty storage or non-empty storage,
   * i.e. both pointer and bytes must be specified
   **/
  inline bool IsValid() const { return this->IsEmpty() || (nullptr != ptr && bytes != 0); }

  /**
   * Returns true only if data is nullptr and no bytes are to be stored
   **/
  inline bool IsEmpty() const { return (0 == bytes) && (nullptr == ptr); }

private:
  void *ptr;// type defined by DataType
  size_t bytes;// how big block is available
  DataType type;// what type is stored
};
}// namespace umpalumpa::data
