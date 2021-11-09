#pragma once

#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/data/data_type.hpp>
#include <libumpalumpa/data/managed_by.hpp>

namespace umpalumpa::data {
class PhysicalDescriptor
{
public:
  explicit PhysicalDescriptor(void *data, size_t b, DataType dataType, ManagedBy m, int node)
    : ptr(data), bytes(b), type(dataType), manager(m), memoryNode(node){};

  explicit PhysicalDescriptor()
    : PhysicalDescriptor(nullptr, 0, DataType::kVoid, ManagedBy::Unknown, 0){};

  inline auto GetBytes() const { return bytes; }

  inline auto GetKBytes() const { return static_cast<float>(bytes) / 1024.f; }

  inline auto GetMBytes() const { return static_cast<float>(bytes) / (1024.f * 1024.f); }

  inline auto GetGBytes() const { return static_cast<float>(bytes) / (1024.f * 1024.f * 1024.f); }

  inline auto GetType() const { return type; }

  inline auto GetManager() const { return manager; }

  inline auto GetMemoryNode() const { return memoryNode; }

  inline void *GetPtr() const { return ptr; }

  auto CopyWithoutData() const { return PhysicalDescriptor(nullptr, 0, type, manager, memoryNode); }

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
  ManagedBy manager;// who is responsible for data
  int memoryNode;// says where exactly is data stored (if supported by Manager)
};
}// namespace umpalumpa::data
