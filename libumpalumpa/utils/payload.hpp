#pragma once

#include <libumpalumpa/data/payload.hpp>
#include <stdexcept>
#include <ostream>
#include <iomanip>
#include <string>
#include <complex>

namespace umpalumpa::utils {

// Data need to be accessible from CPU
template<typename T, typename DT>
void PrivatePrint(std::ostream &out,
  const umpalumpa::data::Payload<T> &p,
  const umpalumpa::data::Size &total,
  const umpalumpa::data::Size &dims,
  const umpalumpa::data::Size &offset)
{

  auto original = out.flags();
  // prepare output formatting
  out << std::fixed << std::setfill(' ') << std::left << std::setprecision(3) << std::showpos;

  auto *data = reinterpret_cast<DT *>(p.GetPtr());
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

// Data need to be accessible from CPU
template<typename T>
void PrintData(std::ostream &out,
  const umpalumpa::data::Payload<T> &p,
  const umpalumpa::data::Size &dims,
  const umpalumpa::data::Size &offset)
{
  auto total = p.info.GetSize();
  switch (p.dataInfo.GetType()) {
  case DataType::kFloat:
    PrivatePrint<T, float>(out, p, total, dims, offset);
    break;
  case DataType::kDouble:
    PrivatePrint<T, double>(out, p, total, dims, offset);
    break;
  case DataType::kComplexFloat:
    PrivatePrint<T, std::complex<float>>(out, p, total, dims, offset);
    break;
  case DataType::kComplexDouble:
    PrivatePrint<T, std::complex<double>>(out, p, total, dims, offset);
    break;
  default:
    throw std::logic_error("Trying to print unprintable type.");
  }
}

template<typename T> void PrintData(std::ostream &out, const umpalumpa::data::Payload<T> &p)
{
  Size offset(0, 0, 0, 0);
  auto dims = p.info.GetSize();
  PrintData(out, p, dims, offset);
}
}// namespace umpalumpa::utils
