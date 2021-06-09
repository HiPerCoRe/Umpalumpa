#pragma once

#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/data/data_type.hpp>

namespace umpalumpa {
namespace data {
  class PhysicalDescriptor
  {
  public:
    explicit PhysicalDescriptor(size_t b, DataType dataType)
      : bytes(b), kbytes(static_cast<float>(b) / 1024),
        Mbytes(static_cast<float>(b) / (1024 * 1024)),
        Gbytes(static_cast<float>(b) / (1024 * 1024 * 1024)), type(dataType){};

    explicit PhysicalDescriptor() : PhysicalDescriptor(0, DataType::kVoid){};

    size_t bytes;
    float kbytes;
    float Mbytes;
    float Gbytes;
    DataType type;

    bool IsValid() const { return true; }

    bool IsEmpty() const { return 0 == bytes; }
  };
}// namespace data
}// namespace umpalumpa