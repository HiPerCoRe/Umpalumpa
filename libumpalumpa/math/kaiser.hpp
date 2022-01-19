#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/blob_order.hpp>
#include <libumpalumpa/math/pi.hpp>
#include <libumpalumpa/math/bessier.hpp>

namespace umpalumpa::math {

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
  return std::numeric_limits<double>::quiet_NaN();
}

}// namespace umpalumpa::math