#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/blob_order.hpp>
#include <libumpalumpa/math/pi.hpp>
#include <libumpalumpa/math/bessier.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::math {

/**
 * This function returns the value of a blob at a given distance from its
 * center (in Universal System units). The distance must be
 * always positive. Remember that a blob is spherically symmetrycal so
 * the only parameter to know the blob value at a point is its distance
 * to the center of the blob. It doesn't matter if this distance is
 * larger than the real blob spatial extension, in this case the function
 * returns 0 as blob value.
 **/
template<typename T>
T KaiserValue(T freq, T blobRadius, T alpha, fourier_reconstruction::BlobOrder order)
{
  using fourier_reconstruction::BlobOrder;
  T w = 0;
  auto rda = freq / blobRadius;
  if (rda <= T(1)) {
    auto rdas = rda * rda;
    auto arg = alpha * std::sqrt(1 - rdas);
    switch (order) {
    case BlobOrder::k0:
      w = static_cast<float>(bessi0(arg) / bessi0(alpha));
      break;
    case BlobOrder::k1:
      w = std::sqrt(1 - rdas);
      if (alpha != 0) w *= static_cast<float>(bessi1(arg) / bessi1(alpha));
      break;
    case BlobOrder::k2:
      w = std::sqrt(1 - rdas);
      w = w * w;
      if (alpha != 0) w *= static_cast<float>(bessi2(arg) / bessi2(alpha));
      break;
    case BlobOrder::k3:
      w = std::sqrt(1 - rdas);
      w = w * w * w;
      if (alpha != 0) w *= static_cast<float>(bessi3(arg) / bessi3(alpha));
      break;
    case BlobOrder::k4:
      w = std::sqrt(1 - rdas);
      w = w * w * w * w;
      if (alpha != 0) w *= static_cast<float>(bessi4(arg) / bessi4(alpha));
      break;
    default:
      spdlog::error("Unsupported order {}", order);
    }
  }
  return w;
}

/**
 * Fourier transform of a blob used for interpolation.
 * This function returns the value of the Fourier transform of the blob
 * at a given frequency (freq). This frequency must be normalized by the
 * sampling rate. For instance, for computing the Fourier Transform of
 * a blob at 1/Ts (Ts in Amstrongs) you must provide the frequency Tm/Ts,
 * where Tm is the sampling rate.
 * The Fourier Transform can be computed only for blobs with order=0. */
template<typename T>
T KaiserFourierValue(T freq, T blobRadius, T alpha, fourier_reconstruction::BlobOrder order)
{
  using fourier_reconstruction::BlobOrder;
  T sigma = std::sqrt(std::abs(
    alpha * alpha - (T(2) * PI<T> * blobRadius * freq) * (T(2) * PI<T> * blobRadius * freq)));
  constexpr T tmp = T(3) / T(2);

  if ((BlobOrder::k0 == order) && (2 * PI<T> * blobRadius * freq <= alpha)) {
    return std::pow(T(2) * PI<T>, tmp) * std::pow(blobRadius, T(3)) * bessi1_5(sigma)
           / (static_cast<float>(bessi0(alpha)) * std::pow(sigma, T(1.5)));
  }
  // not supported
  spdlog::error("Unsupported KaiserFourierValue for order {}", order);
  return std::numeric_limits<double>::quiet_NaN();
}

}// namespace umpalumpa::math