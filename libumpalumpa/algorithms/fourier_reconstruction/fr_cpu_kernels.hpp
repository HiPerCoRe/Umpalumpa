#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/blob_order.hpp>

#include <iostream>

namespace umpalumpa::fourier_reconstruction {

template<bool usePrecomputedInterpolation> struct FR
{

  template<int blobOrder> static void Execute()
  {
    auto ToString = [](auto v) -> std::string {
      if constexpr (std::is_same_v<decltype(v), bool>) return v ? "yes" : "no";
      return std::to_string(v);
    };
    auto Report = [ToString](const std::string &s, auto b) {
      std::cout << s << ": " << ToString(b) << "\n";
    };
    Report("usePrecomputedInterpolation", usePrecomputedInterpolation);
    Report("blobOrder", blobOrder);
  }

  static void Execute(const BlobOrder &order)
  {
    switch (order) {
    case BlobOrder::k0:
      return Execute<0>();
    default:
      return;// not supported
    }
  }
};
}// namespace umpalumpa::fourier_reconstruction