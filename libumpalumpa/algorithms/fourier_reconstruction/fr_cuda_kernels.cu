#include <libumpalumpa/data/size.hpp>
// #include <libumpalumpa/data/point3d.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space.hpp>
// #include <libumpalumpa/algorithms/fourier_reconstruction/constants.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_common_kernels.hpp>
#include <libumpalumpa/math/geometry.hpp>

using umpalumpa::fourier_reconstruction::TraverseSpace;
using umpalumpa::fourier_reconstruction::Constants;
using umpalumpa::fourier_reconstruction::multiply;
using umpalumpa::fourier_reconstruction::clamp;
using umpalumpa::fourier_reconstruction::kaiserValue;
using umpalumpa::fourier_reconstruction::kaiserValueFast;
using umpalumpa::utils::getX;
using umpalumpa::utils::getY;
using umpalumpa::utils::getZ;
using umpalumpa::data::Point3D;

// TODO this in theory can also be in the constant memory, provided it fits
#if SHARED_BLOB_TABLE
__shared__ float BLOB_TABLE[BLOB_TABLE_SIZE_SQRT];// the size of an array must be greater than zero
#endif

#if SHARED_IMG
__shared__ Point3D<float> SHARED_AABB[2];
extern __shared__ float2 IMG[];
#endif

/**
 * Method will map one voxel from the temporal
 * spaces to the given projection and update temporal spaces
 * using the pixel value of the projection.
 */
__device__ void processVoxel(float2 *volume,
  float *weights,
  int x,
  int y,
  int z,
  int xSize,
  int ySize,
  const float2 *__restrict__ FFT,
  const TraverseSpace &space,
  const Constants &gpuC)
{
  Point3D<float> imgPos;
  float wBlob = 1.f;
  float dataWeight = space.weight;

  // transform current point to center
  imgPos.x = x - gpuC.cMaxVolumeIndexX / 2;
  imgPos.y = y - gpuC.cMaxVolumeIndexYZ / 2;
  imgPos.z = z - gpuC.cMaxVolumeIndexYZ / 2;
  if (imgPos.x * imgPos.x + imgPos.y * imgPos.y + imgPos.z * imgPos.z > space.maxDistanceSqr) {
    return;// discard iterations that would access pixel with too high frequency
  }
  // rotate around center
  multiply(space.transformInv, imgPos);
  if (imgPos.x < 0.f)
    return;// reading outside of the image boundary. Z is always correct and Y is checked by the
           // condition above

  // transform back and round
  // just Y coordinate needs adjusting, since X now matches to picture and Z is irrelevant
  int imgX = clamp((int)(imgPos.x + 0.5f), 0, xSize - 1);
  int imgY = clamp((int)(imgPos.y + 0.5f + gpuC.cMaxVolumeIndexYZ / 2), 0, ySize - 1);

  int index3D = z * (gpuC.cMaxVolumeIndexYZ + 1) * (gpuC.cMaxVolumeIndexX + 1)
                + y * (gpuC.cMaxVolumeIndexX + 1) + x;
  int index2D = imgY * xSize + imgX;

  float weight = wBlob * dataWeight;

  // use atomic as two blocks can write to same voxel
  atomicAdd(&volume[index3D].x, FFT[index2D].x * weight);
  atomicAdd(&volume[index3D].y, FFT[index2D].y * weight);
  atomicAdd(&weights[index3D], weight);
}

/**
 * Method will map one voxel from the temporal
 * spaces to the given projection and update temporal spaces
 * using the pixel values of the projection withing the blob distance.
 */
template<int blobOrder, bool useFastKaiser>
__device__ void processVoxelBlob(float2 *tempVolumeGPU,
  float *tempWeightsGPU,
  const int x,
  const int y,
  const int z,
  const int xSize,
  const int ySize,
  const float2 *__restrict__ FFT,
  const TraverseSpace &space,
  const float *blobTableSqrt,
  // const int imgCacheDim FIXME add
  const Constants &gpuC)
{
  Point3D<float> imgPos;
  // transform current point to center
  imgPos.x = x - gpuC.cMaxVolumeIndexX / 2;
  imgPos.y = y - gpuC.cMaxVolumeIndexYZ / 2;
  imgPos.z = z - gpuC.cMaxVolumeIndexYZ / 2;
  if ((imgPos.x * imgPos.x + imgPos.y * imgPos.y + imgPos.z * imgPos.z) > space.maxDistanceSqr) {
    return;// discard iterations that would access pixel with too high frequency
  }
  // rotate around center
  multiply(space.transformInv, imgPos);
  if (imgPos.x < -gpuC.cBlobRadius)
    return;// reading outside of the image boundary. Z is always correct and Y is checked by the
           // condition above
  // transform back just Y coordinate, since X now matches to picture and Z is irrelevant
  imgPos.y += gpuC.cMaxVolumeIndexYZ / 2;

  // check that we don't want to collect data from far far away ...
  float radiusSqr = gpuC.cBlobRadius * gpuC.cBlobRadius;
  float zSqr = imgPos.z * imgPos.z;
  if (zSqr > radiusSqr) return;

  // create blob bounding box
  int minX = ceilf(imgPos.x - gpuC.cBlobRadius);
  int maxX = floorf(imgPos.x + gpuC.cBlobRadius);
  int minY = ceilf(imgPos.y - gpuC.cBlobRadius);
  int maxY = floorf(imgPos.y + gpuC.cBlobRadius);
  minX = fmaxf(minX, 0);
  minY = fmaxf(minY, 0);
  maxX = fminf(maxX, xSize - 1);
  maxY = fminf(maxY, ySize - 1);

  int index3D = z * (gpuC.cMaxVolumeIndexYZ + 1) * (gpuC.cMaxVolumeIndexX + 1)
                + y * (gpuC.cMaxVolumeIndexX + 1) + x;
  float2 vol;
  float w;
  vol.x = vol.y = w = 0.f;
  float dataWeight = space.weight;

  // check which pixel in the vicinity should contribute
  for (int i = minY; i <= maxY; i++) {
    float ySqr = (imgPos.y - i) * (imgPos.y - i);
    float yzSqr = ySqr + zSqr;
    if (yzSqr > radiusSqr) continue;
    for (int j = minX; j <= maxX; j++) {
      float xD = imgPos.x - j;
      float distanceSqr = xD * xD + yzSqr;
      if (distanceSqr > radiusSqr) continue;

#if SHARED_IMG
      int index2D = (i - SHARED_AABB[0].y) * imgCacheDim
                    + (j - SHARED_AABB[0].x);// position in img - offset of the AABB
#else
      int index2D = i * xSize + j;
#endif

#if PRECOMPUTE_BLOB_VAL
      int aux = (int)((distanceSqr * gpuC.cIDeltaSqrt + 0.5f));
#if SHARED_BLOB_TABLE
      float wBlob = BLOB_TABLE[aux];
#else
      float wBlob = blobTableSqrt[aux];
#endif
#else
      float wBlob;
      if (useFastKaiser) {
        wBlob = kaiserValueFast(distanceSqr, gpuC);
      } else {
        wBlob = kaiserValue<blobOrder>(sqrtf(distanceSqr), gpuC.cBlobRadius, gpuC) * gpuC.cIw0;
      }
#endif
      float weight = wBlob * dataWeight;
      w += weight;
#if SHARED_IMG
      vol.x += IMG[index2D].x * weight;
      vol.y += IMG[index2D].y * weight;
#else
      vol.x += FFT[index2D].x * weight;
      vol.y += FFT[index2D].y * weight;
#endif
    }
  }

  // use atomic as two blocks can write to same voxel
  atomicAdd(&tempVolumeGPU[index3D].x, vol.x);
  atomicAdd(&tempVolumeGPU[index3D].y, vol.y);
  atomicAdd(&tempWeightsGPU[index3D], w);
}

/**
 * Method will process one projection image and add result to temporal
 * spaces.
 */
template<int blobOrder, bool useFast, bool useFastKaiser>
__device__ void processProjection(float2 *volume,
  float *weights,
  umpalumpa::data::Size size,
  const TraverseSpace &tSpace,
  const float2 *__restrict__ FFT,
  const float *blobTableSqrt,
  const Constants &gpuC
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
  if (TraverseSpace::Direction::XY == tSpace.dir) {// iterate XY plane
    if (idy >= tSpace.minY && idy <= tSpace.maxY) {
      if (idx >= tSpace.minX && idx <= tSpace.maxX) {
        if (useFast) {
          float hitZ = getZ(float(idx), float(idy), tSpace.unitNormal, tSpace.bottomOrigin);
          int z = (int)(hitZ + 0.5f);// rounding
          processVoxel(volume, weights, idx, idy, z, xSize, ySize, FFT, tSpace, gpuC);
        } else {
          float z1 =
            getZ(float(idx), float(idy), tSpace.unitNormal, tSpace.bottomOrigin);// lower plane
          float z2 =
            getZ(float(idx), float(idy), tSpace.unitNormal, tSpace.topOrigin);// upper plane
          z1 = clamp(z1, 0, gpuC.cMaxVolumeIndexYZ);
          z2 = clamp(z2, 0, gpuC.cMaxVolumeIndexYZ);
          int lower = static_cast<int>(floorf(fminf(z1, z2)));
          int upper = static_cast<int>(ceilf(fmaxf(z1, z2)));
          for (int z = lower; z <= upper; z++) {
            processVoxelBlob<blobOrder, useFastKaiser>(
              volume, weights, idx, idy, z, xSize, ySize, FFT, tSpace, blobTableSqrt, gpuC);
          }
        }
      }
    }
  } else if (TraverseSpace::Direction::XZ == tSpace.dir) {// iterate XZ plane
    if (idy >= tSpace.minZ && idy <= tSpace.maxZ) {// map z -> y
      if (idx >= tSpace.minX && idx <= tSpace.maxX) {
        if (useFast) {
          float hitY = getY(float(idx), float(idy), tSpace.unitNormal, tSpace.bottomOrigin);
          int y = (int)(hitY + 0.5f);// rounding
          processVoxel(volume, weights, idx, y, idy, xSize, ySize, FFT, tSpace, gpuC);
        } else {
          float y1 =
            getY(float(idx), float(idy), tSpace.unitNormal, tSpace.bottomOrigin);// lower plane
          float y2 =
            getY(float(idx), float(idy), tSpace.unitNormal, tSpace.topOrigin);// upper plane
          y1 = clamp(y1, 0, gpuC.cMaxVolumeIndexYZ);
          y2 = clamp(y2, 0, gpuC.cMaxVolumeIndexYZ);
          int lower = static_cast<int>(floorf(fminf(y1, y2)));
          int upper = static_cast<int>(ceilf(fmaxf(y1, y2)));
          for (int y = lower; y <= upper; y++) {
            processVoxelBlob<blobOrder, useFastKaiser>(
              volume, weights, idx, y, idy, xSize, ySize, FFT, tSpace, blobTableSqrt, gpuC);
          }
        }
      }
    }
  } else {// iterate YZ plane
    if (idy >= tSpace.minZ && idy <= tSpace.maxZ) {// map z -> y
      if (idx >= tSpace.minY && idx <= tSpace.maxY) {// map y > x
        if (useFast) {
          float hitX = getX(float(idx), float(idy), tSpace.unitNormal, tSpace.bottomOrigin);
          int x = (int)(hitX + 0.5f);// rounding
          processVoxel(volume, weights, x, idx, idy, xSize, ySize, FFT, tSpace, gpuC);
        } else {
          float x1 =
            getX(float(idx), float(idy), tSpace.unitNormal, tSpace.bottomOrigin);// lower plane
          float x2 =
            getX(float(idx), float(idy), tSpace.unitNormal, tSpace.topOrigin);// upper plane
          x1 = clamp(x1, 0, gpuC.cMaxVolumeIndexX);
          x2 = clamp(x2, 0, gpuC.cMaxVolumeIndexX);
          int lower = static_cast<int>(floorf(fminf(x1, x2)));
          int upper = static_cast<int>(ceilf(fmaxf(x1, x2)));
          for (int x = lower; x <= upper; x++) {
            processVoxelBlob<blobOrder, useFastKaiser>(
              volume, weights, x, idx, idy, xSize, ySize, FFT, tSpace, blobTableSqrt, gpuC);
          }
        }
      }
    }
  }
}


template<int blobOrder, bool fastLateBlobbing, bool useFastKaiser>
__global__ void ProcessKernel(float2 *outVolumeBuffer,
  float *outWeightsBuffer,
  umpalumpa::data::Size size,
  const int traverseSpaceCount,
  const TraverseSpace *traverseSpaces,
  const float2 *FFTs,
  const float *blobTableSqrt,
  Constants gpuC
  // int imgCacheDim FIXME add
)
{

#if SHARED_BLOB_TABLE
  if (!fastLateBlobbing) {
    // copy blob table to shared memory
    volatile int id = threadIdx.y * blockDim.x + threadIdx.x;
    volatile int blockSize = blockDim.x * blockDim.y;
    for (int i = id; i < BLOB_TABLE_SIZE_SQRT; i += blockSize) BLOB_TABLE[i] = blobTableSqrt[i];
    __syncthreads();
  }
#endif

  for (int i = blockIdx.z; i < traverseSpaceCount; i += gridDim.z) {
    const auto &space = traverseSpaces[i];

#if SHARED_IMG
    if (!fastLateBlobbing) {
      // make sure that all threads start at the same time
      // as they can come from previous iteration
      __syncthreads();
      if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        // first thread calculates which part of the image should be shared
        calculateAABB(&space, SHARED_AABB);
      }
      __syncthreads();
      // check if the block will have to copy data from image
      if (isWithin(SHARED_AABB, fftSizeX, fftSizeY)) {
        // all threads copy image data to shared memory
        copyImgToCache(
          IMG, SHARED_AABB, FFTs, fftSizeX, fftSizeY, space.projectionIndex, imgCacheDim);
        __syncthreads();
      } else {
        continue;// whole block can exit, as it's not reading from image
      }
    }
#endif


    processProjection<blobOrder, fastLateBlobbing, useFastKaiser>(outVolumeBuffer,
      outWeightsBuffer,
      size,
      space,
      FFTs + size.single * space.projectionIndex,
      blobTableSqrt,
      gpuC
      // imgCacheDim
    );
    __syncthreads();// sync threads to avoid write after read problems
  }
}