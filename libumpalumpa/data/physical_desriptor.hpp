#pragma once

#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/data/data_type.hpp>

namespace umpalumpa {
namespace data {
  class PhysicalDescriptor
  {
  public:
    explicit PhysicalDescriptor(size_t b, DataType dataType)
      : bytes(b), kbytes(static_cast<float>(b) / 1024), Mbytes(static_cast<float>(b) / (1024 * 1024)),
        Gbytes(static_cast<float>(b) / (1024 * 1024 * 1024)), type(dataType){};

    const size_t bytes;
    const float kbytes;
    const float Mbytes;
    const float Gbytes;
    const DataType type;
  };
}// namespace data
}// namespace umpalumpa