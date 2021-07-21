#pragma once

#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/data/data_type.hpp>

namespace umpalumpa {
namespace data {
  class PhysicalDescriptor
  {
  public:
  // FIXME this should hold the data pointer, Payload should call some getter from here to get them
    explicit PhysicalDescriptor(size_t b, DataType dataType)
      : bytes(b), kbytes(static_cast<float>(b) / 1024),
        Mbytes(static_cast<float>(b) / (1024 * 1024)),
        Gbytes(static_cast<float>(b) / (1024 * 1024 * 1024)), type(dataType){};

    explicit PhysicalDescriptor() : PhysicalDescriptor(0, DataType::kVoid){};

    // these shouold be private + getters
    size_t bytes;
    // fixme it would be cheaper to compute these on demand
    float kbytes;
    float Mbytes;
    float Gbytes;
    DataType type;

    bool IsValid() const { return true; }

    bool IsEmpty() const { return 0 == bytes; }
  };
}// namespace data
}// namespace umpalumpa