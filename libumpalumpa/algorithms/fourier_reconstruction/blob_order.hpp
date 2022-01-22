#pragma once

namespace umpalumpa::fourier_reconstruction {

enum class BlobOrder { k0, k1, k2, k3, k4 };

[[maybe_unused]] static int ToInt(BlobOrder o)
{
  switch (o) {
  case BlobOrder::k0:
    return 0;
  case BlobOrder::k1:
    return 1;
  case BlobOrder::k2:
    return 2;
  case BlobOrder::k3:
    return 3;
  case BlobOrder::k4:
    return 4;
  default:
    return -1;
  }
}

}// namespace umpalumpa::fourier_reconstruction