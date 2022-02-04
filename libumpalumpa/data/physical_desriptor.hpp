#pragma once

#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/data/data_type.hpp>
#include <libumpalumpa/data/managed_by.hpp>
#include <type_traits>

namespace umpalumpa::data {
/**
 * Class describing actual storage of some data chunk
 **/
class PhysicalDescriptor
{
public:
  explicit PhysicalDescriptor(void *data, size_t b, DataType dataType, ManagedBy m, void *h)
    : ptr(data), bytes(b), type(dataType), manager(m), handle(h)
  {
    static_assert(std::is_move_constructible<PhysicalDescriptor>::value);
  };

  auto GetBytes() const { return bytes; }

  auto GetKBytes() const { return static_cast<float>(bytes) / 1024.f; }

  auto GetMBytes() const { return static_cast<float>(bytes) / (1024.f * 1024.f); }

  auto GetGBytes() const { return static_cast<float>(bytes) / (1024.f * 1024.f * 1024.f); }

  auto GetType() const { return type; }

  auto GetManager() const { return manager; }

  auto *GetHandle() const { return handle; }

  void *GetPtr() const { return ptr; }

  bool IsPinned() const { return pinned; };

  auto CopyWithoutData() const { return PhysicalDescriptor(nullptr, 0, type, manager, nullptr); }

  auto CopyWithPtr(void *p) const { return PhysicalDescriptor(p, bytes, type, manager, handle); }

  void SetPinned(bool isPinned) { pinned = isPinned; };

  /**
   * Descriptor is valid if it describes empty storage or non-empty storage,
   * i.e. both pointer and bytes must be specified
   **/
  bool IsValid() const { return this->IsEmpty() || (nullptr != ptr && bytes != 0); }

  /**
   * Returns true only if data is nullptr and no bytes are to be stored
   **/
  bool IsEmpty() const { return (0 == bytes) && (nullptr == ptr); }

  PhysicalDescriptor(PhysicalDescriptor &&) = default;

  bool operator==(const PhysicalDescriptor &o) const
  {
    return ptr == o.ptr && bytes == o.bytes && type == o.type && manager == o.manager
           && handle == o.handle;
  }

private:
  // Prevent copying of this instance (to avoid accidental handle copy)
  PhysicalDescriptor(const PhysicalDescriptor &) = default;
  constexpr PhysicalDescriptor &operator=(const PhysicalDescriptor &) = default;
  void *ptr;// type defined by DataType
  size_t bytes;// how big block is available
  DataType type;// what type is stored
  ManagedBy manager;// who is responsible for data
  void *handle;// handle used by the manager (if any)
  bool pinned = false;// memory has been pinned in RAM
};
}// namespace umpalumpa::data
