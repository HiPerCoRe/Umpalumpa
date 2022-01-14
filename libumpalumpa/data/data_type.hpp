#pragma once

#include <typeindex>

namespace umpalumpa::data {

class DataType
{
public:
  DataType(const std::type_info &info, size_t bytes) : index(info), size(bytes) {}

  template<typename T> static DataType Get() { return DataType(typeid(T), sizeof(T)); }

  auto GetSize() const { return size; }

  auto GetIndex() const { return index; }

  bool operator==(const DataType &o) const
  {
    return index == o.index;// and size has to be the same
  }

  template<typename T> bool Is() const { return index == std::type_index(typeid(T)); }

private:
  const std::type_index index;
  const std::size_t size;
};

template<> inline DataType DataType::Get<void>()
{
  // to avoid warning: invalid application of ‘sizeof’ to a void type [-Wpointer-arith]
  return DataType(typeid(void), 0);
}
}// namespace umpalumpa::data