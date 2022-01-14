#pragma once

#include <complex>
#include <algorithm>// std::clamp
#include <libumpalumpa/algorithms/fourier_reconstruction/blob_order.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space.hpp>
#include <libumpalumpa/utils/geometry.hpp>

#include <iostream>

namespace umpalumpa::fourier_reconstruction {

struct Constants
{
  int cMaxVolumeIndexX;
  int cMaxVolumeIndexYZ;
  float cBlobRadius;
  float cOneOverBlobRadiusSqr;
  float cBlobAlpha;
  float cIw0;
  float cIDeltaSqrt;
  float cOneOverBessiOrderAlpha;
};

template<bool useFast, bool usePrecomputedInterpolation, bool useFastKaiser> class FR
{
public:
  template<int blobOrder, typename T>
  static void Execute(std::complex<T> *__restrict__ volume,
    T *__restrict__ weights,
    const int xSize,
    const int ySize,
    const std::complex<T> *__restrict__ FFT,
    const TraverseSpace *const __restrict__ tSpace,
    const T *__restrict__ blobTableSqrt,
    const Constants &constants)
  {
    switch (tSpace->dir) {
    case TraverseSpace::Direction::XY: {
      for (int idy = tSpace->minY; idy <= tSpace->maxY; idy++) {
        const T idyT = static_cast<T>(idy);
        for (int idx = tSpace->minX; idx <= tSpace->maxX; idx++) {
          const T idxT = static_cast<T>(idx);
          if (useFast) {
            // float hitZ = getZ(idxT, idyT, tSpace->unitNormal, tSpace->bottomOrigin);
            // int z = static_cast<int>(hitZ + 0.5f);// rounding
            // processVoxelCPU(volume, weights, idx, idy, z, xSize, ySize, FFT, tSpace);
          } else {
            float z1 = getZ(idxT, idyT, tSpace->unitNormal, tSpace->bottomOrigin);// lower plane
            float z2 = getZ(idxT, idyT, tSpace->unitNormal, tSpace->topOrigin);// upper plane
            z1 = std::clamp(z1, 0.f, static_cast<float>(constants.cMaxVolumeIndexYZ));
            z2 = std::clamp(z2, 0.f, static_cast<float>(constants.cMaxVolumeIndexYZ));
            int lower = static_cast<int>(floorf(fminf(z1, z2)));
            int upper = static_cast<int>(ceilf(fmaxf(z1, z2)));
            for (int z = lower; z <= upper; z++) {
              // processVoxelBlobCPU<blobOrder, useFastKaiser, usePrecomputedInterpolation>(
              //   volume, weights, idx, idy, z, xSize, ySize, FFT, tSpace, blobTableSqrt);
            }
          }
        }
      }
    } break;
    case TraverseSpace::Direction::XZ: {
      for (int idy = tSpace->minZ; idy <= tSpace->maxZ; idy++) {// map z -> y
        const T idyT = static_cast<T>(idy);
        for (int idx = tSpace->minX; idx <= tSpace->maxX; idx++) {
          const T idxT = static_cast<T>(idx);
          if (useFast) {
            // float hitY = getY<T>(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);
            // int y = static_cast<int>(hitY + 0.5f);// rounding
            // processVoxelCPU(volume, weights, idx, y, idy, xSize, ySize, FFT, tSpace);
          } else {
            float y1 = getY(idxT, idyT, tSpace->unitNormal, tSpace->bottomOrigin);// lower plane
            float y2 = getY(idxT, idyT, tSpace->unitNormal, tSpace->topOrigin);// upper plane
            y1 = std::clamp(y1, 0.f, static_cast<float>(constants.cMaxVolumeIndexYZ));
            y2 = std::clamp(y2, 0.f, static_cast<float>(constants.cMaxVolumeIndexYZ));
            int lower = static_cast<int>(floorf(fminf(y1, y2)));
            int upper = static_cast<int>(ceilf(fmaxf(y1, y2)));
            for (int y = lower; y <= upper; y++) {
              // processVoxelBlobCPU<blobOrder, useFastKaiser, usePrecomputedInterpolation>(
              //   volume, weights, idx, y, idy, xSize, ySize, FFT, tSpace, blobTableSqrt);
            }
          }
        }
      }
    } break;
    case TraverseSpace::Direction::YZ: {
      for (int idy = tSpace->minZ; idy <= tSpace->maxZ; idy++) {// map z -> y
        const T idyT = static_cast<T>(idy);
        for (int idx = tSpace->minY; idx <= tSpace->maxY; idx++) {// map y > x
         const T idxT = static_cast<T>(idx);
          if (useFast) {
            // float hitX = getX<T>(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);
            // int x = static_cast<int>(hitX + 0.5f);// rounding
            // processVoxelCPU(volume, weights, x, idx, idy, xSize, ySize, FFT, tSpace);
          } else {
            float x1 = getX(idxT, idyT, tSpace->unitNormal, tSpace->bottomOrigin);// lower plane
            float x2 = getX(idxT, idyT, tSpace->unitNormal, tSpace->topOrigin);// upper plane
            x1 = std::clamp(x1, 0.f, static_cast<float>(constants.cMaxVolumeIndexX));
            x2 = std::clamp(x2, 0.f, static_cast<float>(constants.cMaxVolumeIndexX));
            int lower = static_cast<int>(floorf(fminf(x1, x2)));
            int upper = static_cast<int>(ceilf(fmaxf(x1, x2)));
            for (int x = lower; x <= upper; x++) {
              // processVoxelBlobCPU<blobOrder, useFastKaiser, usePrecomputedInterpolation>(
              //   volume, weights, x, idx, idy, xSize, ySize, FFT, tSpace, blobTableSqrt);
            }
          }
        }
      }
    } break;
    }

    auto ToString = [](auto v) -> std::string {
      if constexpr (std::is_same_v<decltype(v), bool>) return v ? "yes" : "no";
      return std::to_string(v);
    };
    auto Report = [ToString](const std::string &s, auto b) {
      std::cout << s << ": " << ToString(b) << "\n";
    };
    Report("usePrecomputedInterpolation", usePrecomputedInterpolation);
    Report("blobOrder", blobOrder);
    Report("useFastKaiser", useFastKaiser);
    Report("useFast", useFast);
    std::cout << "volume: " << volume << "\n";
    std::cout << "weights: " << weights << "\n";
    std::cout << "xSize: " << xSize << "\n";
    std::cout << "ySize: " << ySize << "\n";
    std::cout << "FFT: " << FFT << "\n";
    std::cout << "tSpace: " << tSpace << "\n";
    std::cout << "blobTableSqrt: " << blobTableSqrt << "\n";
    std::cout << "constants: " << &constants << "\n";
  }

  template<typename T>
  static void Execute(const BlobOrder &order,
    std::complex<T> *__restrict__ volume,
    T *__restrict__ weights,
    const int xSize,
    const int ySize,
    const std::complex<T> *__restrict__ FFT,
    const TraverseSpace *const __restrict__ tSpace,
    const T *__restrict__ blobTableSqrt,
    const Constants &constants)
  {
    switch (order) {
    case BlobOrder::k0:
      return Execute<0>(volume, weights, xSize, ySize, FFT, tSpace, blobTableSqrt, constants);
    // case BlobOrder::k1:
    //   return Execute<1>(volume, weights, xSize, ySize, FFT, tSpace, blobTableSqrt);
    // case BlobOrder::k2:
    //   return Execute<2>(volume, weights, xSize, ySize, FFT, tSpace, blobTableSqrt);
    // case BlobOrder::k3:
    //   return Execute<3>(volume, weights, xSize, ySize, FFT, tSpace, blobTableSqrt);
    // case BlobOrder::k4:
    //   return Execute<4>(volume, weights, xSize, ySize, FFT, tSpace, blobTableSqrt);
    default:
      return;// not supported
    }
  }
};
}// namespace umpalumpa::fourier_reconstruction