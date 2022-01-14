#pragma once

#include <complex>
#include <algorithm>// std::clamp
#include <libumpalumpa/algorithms/fourier_reconstruction/blob_order.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_common_kernels.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/constants.hpp>
#include <libumpalumpa/utils/geometry.hpp>
#include <libumpalumpa/utils/math.hpp>

#include <iostream>

namespace umpalumpa::fourier_reconstruction {

template<bool useFast, bool usePrecomputedInterpolation, bool useFastKaiser> class FR
{
public:
  static void ProcessVoxel(std::complex<float> *const __restrict__ volume,
    float *const __restrict__ weights,
    const int x,
    const int y,
    const int z,
    const int xSize,
    const int ySize,
    const std::complex<float> *const __restrict__ FFT,
    const TraverseSpace *const __restrict__ space,
    const Constants &constants)
  {
    float wBlob = 1.f;

    float dataWeight = space->weight;

    // transform current point to center
    data::Point3D<float> imgPos;
    imgPos.x = static_cast<float>(x - constants.cMaxVolumeIndexX / 2);
    imgPos.y = static_cast<float>(y - constants.cMaxVolumeIndexYZ / 2);
    imgPos.z = static_cast<float>(z - constants.cMaxVolumeIndexYZ / 2);
    if (imgPos.x * imgPos.x + imgPos.y * imgPos.y + imgPos.z * imgPos.z > space->maxDistanceSqr) {
      return;// discard iterations that would access pixel with too high frequency
    }
    // rotate around center
    multiply(space->transformInv, imgPos);
    if (imgPos.x < 0.f)
      return;// reading outside of the image boundary. Z is always correct and Y is checked by the
             // condition above

    // transform back and round
    // just Y coordinate needs adjusting, since X now matches to picture and Z is irrelevant
    int imgX = std::clamp(static_cast<int>(imgPos.x + 0.5f), 0, xSize - 1);
    int imgY = std::clamp(
      static_cast<int>(imgPos.y + 0.5f + static_cast<float>(constants.cMaxVolumeIndexYZ / 2)),
      0,
      ySize - 1);

    int index3D = z * (constants.cMaxVolumeIndexYZ + 1) * (constants.cMaxVolumeIndexX + 1)
                  + y * (constants.cMaxVolumeIndexX + 1) + x;
    int index2D = imgY * xSize + imgX;

    float w = wBlob * dataWeight;

    // use atomic as two blocks can write to same voxel
    utils::AtomicAddFloat(
      reinterpret_cast<float *>(volume) + (index3D * 2 + 0), FFT[index2D].real() * w);
    utils::AtomicAddFloat(
      reinterpret_cast<float *>(volume) + (index3D * 2 + 1), FFT[index2D].imag() * w);
    utils::AtomicAddFloat(reinterpret_cast<float *>(weights) + index3D, w);
  }

  template<int blobOrder>
  static void ProcessVoxelBlob(std::complex<float> *const __restrict__ volume,
    float *const __restrict__ weights,
    const int x,
    const int y,
    const int z,
    const int xSize,
    const int ySize,
    const std::complex<float> *const __restrict__ FFT,
    const TraverseSpace *const __restrict__ space,
    const float *__restrict__ blobTableSqrt,
    const Constants &constants)
  {
    data::Point3D<float> imgPos;
    // transform current point to center
    imgPos.x = static_cast<float>(x - constants.cMaxVolumeIndexX / 2);
    imgPos.y = static_cast<float>(y - constants.cMaxVolumeIndexYZ / 2);
    imgPos.z = static_cast<float>(z - constants.cMaxVolumeIndexYZ / 2);
    if ((imgPos.x * imgPos.x + imgPos.y * imgPos.y + imgPos.z * imgPos.z) > space->maxDistanceSqr) {
      return;// discard iterations that would access pixel with too high frequency
    }
    // rotate around center
    multiply(space->transformInv, imgPos);
    if (imgPos.x < -constants.cBlobRadius)
      return;// reading outside of the image boundary. Z is always correct and Y is checked by the
             // condition above
    // transform back just Y coordinate, since X now matches to picture and Z is irrelevant
    imgPos.y += static_cast<float>(constants.cMaxVolumeIndexYZ / 2);

    // check that we don't want to collect data from far far away ...
    float radiusSqr = constants.cBlobRadius * constants.cBlobRadius;
    float zSqr = imgPos.z * imgPos.z;
    if (zSqr > radiusSqr) return;

    // create blob bounding box
    int minX = static_cast<int>(ceilf(imgPos.x - constants.cBlobRadius));
    int maxX = static_cast<int>(floorf(imgPos.x + constants.cBlobRadius));
    int minY = static_cast<int>(ceilf(imgPos.y - constants.cBlobRadius));
    int maxY = static_cast<int>(floorf(imgPos.y + constants.cBlobRadius));
    minX = std::max(minX, 0);
    minY = std::max(minY, 0);
    maxX = std::min(maxX, xSize - 1);
    maxY = std::min(maxY, ySize - 1);

    int index3D = z * (constants.cMaxVolumeIndexYZ + 1) * (constants.cMaxVolumeIndexX + 1)
                  + y * (constants.cMaxVolumeIndexX + 1) + x;
    float w = 0.f;
    std::complex<float> vol = { 0.f, 0.f };
    float dataWeight = space->weight;

    // check which pixel in the vicinity should contribute
    for (int i = minY; i <= maxY; i++) {
      float ySqr = (imgPos.y - static_cast<float>(i)) * (imgPos.y - static_cast<float>(i));
      float yzSqr = ySqr + zSqr;
      if (yzSqr > radiusSqr) continue;
      for (int j = minX; j <= maxX; j++) {
        float xD = imgPos.x - static_cast<float>(j);
        float distanceSqr = xD * xD + yzSqr;
        if (distanceSqr > radiusSqr) continue;

        int index2D = i * xSize + j;

        float wBlob;
        if (usePrecomputedInterpolation) {
          int aux = static_cast<int>((distanceSqr * constants.cIDeltaSqrt + 0.5f));
          wBlob = blobTableSqrt[aux];
        } else if (useFastKaiser) {
          wBlob = kaiserValueFast(distanceSqr, constants);
        } else {
          wBlob = kaiserValue<blobOrder>(sqrtf(distanceSqr), constants.cBlobRadius, constants)
                  * constants.cIw0;
        }

        float weight = wBlob * dataWeight;
        w += weight;
        vol += FFT[index2D] * weight;
      }
    }

    utils::AtomicAddFloat(reinterpret_cast<float *>(volume) + (index3D * 2 + 0), vol.real());
    utils::AtomicAddFloat(reinterpret_cast<float *>(volume) + (index3D * 2 + 1), vol.imag());
    utils::AtomicAddFloat(reinterpret_cast<float *>(weights) + index3D, w);
  }

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
    using namespace utils;

    switch (tSpace->dir) {
    case TraverseSpace::Direction::XY: {
      for (int idy = tSpace->minY; idy <= tSpace->maxY; idy++) {
        const T idyT = static_cast<T>(idy);
        for (int idx = tSpace->minX; idx <= tSpace->maxX; idx++) {
          const T idxT = static_cast<T>(idx);
          if (useFast) {
            float hitZ = getZ(idxT, idyT, tSpace->unitNormal, tSpace->bottomOrigin);
            int z = static_cast<int>(hitZ + 0.5f);// rounding
            ProcessVoxel(volume, weights, idx, idy, z, xSize, ySize, FFT, tSpace, constants);
          } else {
            float z1 = getZ(idxT, idyT, tSpace->unitNormal, tSpace->bottomOrigin);// lower plane
            float z2 = getZ(idxT, idyT, tSpace->unitNormal, tSpace->topOrigin);// upper plane
            z1 = std::clamp(z1, 0.f, static_cast<float>(constants.cMaxVolumeIndexYZ));
            z2 = std::clamp(z2, 0.f, static_cast<float>(constants.cMaxVolumeIndexYZ));
            int lower = static_cast<int>(floorf(fminf(z1, z2)));
            int upper = static_cast<int>(ceilf(fmaxf(z1, z2)));
            for (int z = lower; z <= upper; z++) {
              ProcessVoxelBlob<blobOrder>(
                volume, weights, idx, idy, z, xSize, ySize, FFT, tSpace, blobTableSqrt, constants);
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
            float hitY = getY<T>(idxT, idyT, tSpace->unitNormal, tSpace->bottomOrigin);
            int y = static_cast<int>(hitY + 0.5f);// rounding
            ProcessVoxel(volume, weights, idx, y, idy, xSize, ySize, FFT, tSpace, constants);
          } else {
            float y1 = getY(idxT, idyT, tSpace->unitNormal, tSpace->bottomOrigin);// lower plane
            float y2 = getY(idxT, idyT, tSpace->unitNormal, tSpace->topOrigin);// upper plane
            y1 = std::clamp(y1, 0.f, static_cast<float>(constants.cMaxVolumeIndexYZ));
            y2 = std::clamp(y2, 0.f, static_cast<float>(constants.cMaxVolumeIndexYZ));
            int lower = static_cast<int>(floorf(fminf(y1, y2)));
            int upper = static_cast<int>(ceilf(fmaxf(y1, y2)));
            for (int y = lower; y <= upper; y++) {
              ProcessVoxelBlob<blobOrder>(
                volume, weights, idx, y, idy, xSize, ySize, FFT, tSpace, blobTableSqrt, constants);
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
            float hitX = getX<T>(idxT, idyT, tSpace->unitNormal, tSpace->bottomOrigin);
            int x = static_cast<int>(hitX + 0.5f);// rounding
            ProcessVoxel(volume, weights, x, idx, idy, xSize, ySize, FFT, tSpace, constants);
          } else {
            float x1 = getX(idxT, idyT, tSpace->unitNormal, tSpace->bottomOrigin);// lower plane
            float x2 = getX(idxT, idyT, tSpace->unitNormal, tSpace->topOrigin);// upper plane
            x1 = std::clamp(x1, 0.f, static_cast<float>(constants.cMaxVolumeIndexX));
            x2 = std::clamp(x2, 0.f, static_cast<float>(constants.cMaxVolumeIndexX));
            int lower = static_cast<int>(floorf(fminf(x1, x2)));
            int upper = static_cast<int>(ceilf(fmaxf(x1, x2)));
            for (int x = lower; x <= upper; x++) {
              ProcessVoxelBlob<blobOrder>(
                volume, weights, x, idx, idy, xSize, ySize, FFT, tSpace, blobTableSqrt, constants);
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
    case BlobOrder::k1:
      return Execute<1>(volume, weights, xSize, ySize, FFT, tSpace, blobTableSqrt, constants);
    case BlobOrder::k2:
      return Execute<2>(volume, weights, xSize, ySize, FFT, tSpace, blobTableSqrt, constants);
    case BlobOrder::k3:
      return Execute<3>(volume, weights, xSize, ySize, FFT, tSpace, blobTableSqrt, constants);
    case BlobOrder::k4:
      return Execute<4>(volume, weights, xSize, ySize, FFT, tSpace, blobTableSqrt, constants);
    default:
      return;// not supported
    }
  }
};
}// namespace umpalumpa::fourier_reconstruction