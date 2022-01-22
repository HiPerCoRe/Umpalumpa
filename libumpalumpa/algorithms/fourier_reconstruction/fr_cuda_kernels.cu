#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space.hpp>

/**
 * Method will process one projection image and add result to temporal
 * spaces.
 */
template<int blobOrder, bool useFast, bool useFastKaiser>
__global__ void ProcessProjection(float2 *tempVolumeGPU,
  float *tempWeightsGPU,
  umpalumpa::data::Size size,
  const float2 *__restrict__ FFT,
  const umpalumpa::fourier_reconstruction::TraverseSpace *const tSpace,
  const float *blobTableSqrt
//   const int imgCacheDim // FIXME add
  )
{
  const int xSize = size.x;
  const int ySize = size.y;
  // map thread to each (2D) voxel
#if TILE > 1
  int id = threadIdx.y * blockDim.x + threadIdx.x;
  int tidX = threadIdx.x % TILE + (id / (blockDim.y * TILE)) * TILE;
  int tidY = (id / TILE) % blockDim.y;
  int idx = blockIdx.x * blockDim.x + tidX;
  int idy = blockIdx.y * blockDim.y + tidY;
#else
  // map thread to each (2D) voxel
  volatile int idx = blockIdx.x * blockDim.x + threadIdx.x;
  volatile int idy = blockIdx.y * blockDim.y + threadIdx.y;
#endif

  if (idx == 0 || idy == 0) { printf("yay\n"); }

  //   if (tSpace->XY == tSpace->dir) {// iterate XY plane
  //     if (idy >= tSpace->minY && idy <= tSpace->maxY) {
  //       if (idx >= tSpace->minX && idx <= tSpace->maxX) {
  //         if (useFast) {
  //           float hitZ = getZ(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);
  //           int z = (int)(hitZ + 0.5f);// rounding
  //           processVoxel(tempVolumeGPU, tempWeightsGPU, idx, idy, z, xSize, ySize, FFT, tSpace);
  //         } else {
  //           float z1 = getZ(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);// lower plane
  //           float z2 = getZ(idx, idy, tSpace->unitNormal, tSpace->topOrigin);// upper plane
  //           z1 = clamp(z1, 0, gpuC.cMaxVolumeIndexYZ);
  //           z2 = clamp(z2, 0, gpuC.cMaxVolumeIndexYZ);
  //           int lower = static_cast<int>(floorf(fminf(z1, z2)));
  //           int upper = static_cast<int>(ceilf(fmaxf(z1, z2)));
  //           for (int z = lower; z <= upper; z++) {
  //             processVoxelBlob<blobOrder, useFastKaiser>(tempVolumeGPU,
  //               tempWeightsGPU,
  //               idx,
  //               idy,
  //               z,
  //               xSize,
  //               ySize,
  //               FFT,
  //               tSpace,
  //               blobTableSqrt,
  //               imgCacheDim);
  //           }
  //         }
  //       }
  //     }
  //   } else if (tSpace->XZ == tSpace->dir) {// iterate XZ plane
  //     if (idy >= tSpace->minZ && idy <= tSpace->maxZ) {// map z -> y
  //       if (idx >= tSpace->minX && idx <= tSpace->maxX) {
  //         if (useFast) {
  //           float hitY = getY(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);
  //           int y = (int)(hitY + 0.5f);// rounding
  //           processVoxel(tempVolumeGPU, tempWeightsGPU, idx, y, idy, xSize, ySize, FFT, tSpace);
  //         } else {
  //           float y1 = getY(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);// lower plane
  //           float y2 = getY(idx, idy, tSpace->unitNormal, tSpace->topOrigin);// upper plane
  //           y1 = clamp(y1, 0, gpuC.cMaxVolumeIndexYZ);
  //           y2 = clamp(y2, 0, gpuC.cMaxVolumeIndexYZ);
  //           int lower = static_cast<int>(floorf(fminf(y1, y2)));
  //           int upper = static_cast<int>(ceilf(fmaxf(y1, y2)));
  //           for (int y = lower; y <= upper; y++) {
  //             processVoxelBlob<blobOrder, useFastKaiser>(tempVolumeGPU,
  //               tempWeightsGPU,
  //               idx,
  //               y,
  //               idy,
  //               xSize,
  //               ySize,
  //               FFT,
  //               tSpace,
  //               blobTableSqrt,
  //               imgCacheDim);
  //           }
  //         }
  //       }
  //     }
  //   } else {// iterate YZ plane
  //     if (idy >= tSpace->minZ && idy <= tSpace->maxZ) {// map z -> y
  //       if (idx >= tSpace->minY && idx <= tSpace->maxY) {// map y > x
  //         if (useFast) {
  //           float hitX = getX(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);
  //           int x = (int)(hitX + 0.5f);// rounding
  //           processVoxel(tempVolumeGPU, tempWeightsGPU, x, idx, idy, xSize, ySize, FFT, tSpace);
  //         } else {
  //           float x1 = getX(idx, idy, tSpace->unitNormal, tSpace->bottomOrigin);// lower plane
  //           float x2 = getX(idx, idy, tSpace->unitNormal, tSpace->topOrigin);// upper plane
  //           x1 = clamp(x1, 0, gpuC.cMaxVolumeIndexX);
  //           x2 = clamp(x2, 0, gpuC.cMaxVolumeIndexX);
  //           int lower = static_cast<int>(floorf(fminf(x1, x2)));
  //           int upper = static_cast<int>(ceilf(fmaxf(x1, x2)));
  //           for (int x = lower; x <= upper; x++) {
  //             processVoxelBlob<blobOrder, useFastKaiser>(tempVolumeGPU,
  //               tempWeightsGPU,
  //               x,
  //               idx,
  //               idy,
  //               xSize,
  //               ySize,
  //               FFT,
  //               tSpace,
  //               blobTableSqrt,
  //               imgCacheDim);
  //           }
  //         }
  //       }
  //     }
  //   }
  __syncthreads(); // sync threads to avoid write after read problems
}