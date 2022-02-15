#pragma once

#include <iostream>
#include <typeindex>

namespace umpalumpa::data {

class DataType
{
public:
  DataType(const std::type_info &info, size_t bytes) : hashCode(info.hash_code()), size(bytes) {}
  // DataType(const DataType &) = default;
  // DataType &operator=(const DataType &) = default;

  template<typename T> static DataType Get() { return DataType(typeid(T), sizeof(T)); }

  auto GetSize() const { return size; }

  // auto GetIndex() const { return index; }

  bool operator==(const DataType &o) const
  {
    return hashCode == o.hashCode;// and size has to be the same
  }

  template<typename T> bool Is() const { return hashCode == typeid(T).hash_code(); }

  void Serialize(std::ostream &out) const { out << hashCode << ' ' << size << '\n'; }
  static auto Deserialize(std::istream &in)
  {
    size_t h, s;
    in >> h >> s;
    return DataType(h, s);
  }

private:
  // NOTE because of serialization we need to have this class without the std::type_index
  // it doesn't have any specific usage apart from checking type equality, which can be achieved via
  // std::type_info::hash_code
  // const std::type_index index;
  size_t hashCode;
  size_t size;

  DataType(size_t h, size_t s) : hashCode(h), size(s) {}
};

template<> inline DataType DataType::Get<void>()
{
  // to avoid warning: invalid application of ‘sizeof’ to a void type [-Wpointer-arith]
  return DataType(typeid(void), 0);
}
}// namespace umpalumpa::data
