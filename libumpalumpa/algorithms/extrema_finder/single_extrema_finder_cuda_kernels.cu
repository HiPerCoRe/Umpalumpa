#pragma once
#include <libumpalumpa/data/size.hpp>

#ifndef blockSize
#define blockSize (blockSizeX * blockSizeY)
#endif

template<typename T, typename T2>//, typename C>
__device__ void update(// const C &comp,
  T2 &orig,
  const T *__restrict data,
  unsigned index)
{
  T tmp = data[index];
  //   if (comp(tmp, orig.x)) {
  if (tmp > orig.x) {
    orig.x = tmp;
    orig.y = (T)index;
  }
}

template<typename T>//, typename C>
__device__ T update(// const C &comp,
  T &orig,
  T &cand)
{
  //   if (comp(cand.x, orig.x)) {
  if (cand.x > orig.x) {
    orig.x = cand.x;
    orig.y = cand.y;
  }
  return orig;
}

template<typename T>//, unsigned blockSize>//, typename C>
__device__ void findUniversalInSharedMem(
  // const C &comp,
  T &ldata,
  unsigned int tid)
{
  // we have read all data, one of the thread knows the result
  __shared__ T sdata[blockSize];
  sdata[tid] = ldata;
  __syncthreads();// wait till all threads store their data
  // reduce
#pragma unroll
  for (unsigned counter = blockSize / 2; counter >= 32; counter /= 2) {
    if (tid < counter) {
      sdata[tid] = update(// comp,
        ldata,
        sdata[tid + counter]);
    }
    __syncthreads();
  }
  // manually unwrap last warp for better performance
  // many of these blocks will be optimized out by compiler based on template
  if ((blockSize >= 32) && (tid < 16)) {
    sdata[tid] = update(// comp,
      ldata,
      sdata[tid + 16]);
  }
  __syncthreads();
  if ((blockSize >= 16) && (tid < 8)) {
    sdata[tid] = update(// comp,
      ldata,
      sdata[tid + 8]);
  }
  __syncthreads();
  if ((blockSize >= 8) && (tid < 4)) {
    sdata[tid] = update(// comp,
      ldata,
      sdata[tid + 4]);
  }
  __syncthreads();
  if ((blockSize >= 4) && (tid < 2)) {
    sdata[tid] = update(// comp,
      ldata,
      sdata[tid + 2]);
  }
  __syncthreads();
  if ((blockSize >= 2) && (tid < 1)) {
    sdata[tid] = update(// comp,
      ldata,
      sdata[tid + 1]);
  }
  __syncthreads();
}

// template <typename T, typename T2, unsigned blockSize, typename C>
// __device__
// void findMax1D(
//         const C &comp,
//         T startVal,
//         const T * __restrict__ in,
//         float * __restrict__ outPos,
//         T * __restrict__ outVal,
//         unsigned samples)
// {
extern "C" __global__ void findMax1D(
  // const C &comp,
  // T startVal,
  float *in,
  // float * __restrict__ outPos,
  float *outVal,
  unsigned samples)
{
  // auto comp = [] (float l, float r) { return l > r; };
  // one block processes one signal
  // map each thread to some sample of the signal
  unsigned int tid = threadIdx.x;
  unsigned int signal = blockIdx.x;

  // if (signal ==0 && tid == 0) {
  //   for (int i = 0; i < 30; i++) {
  //     printf("%f ", in[i]);
  //   }
  // }

  // load data from global memory
  const float *data = in + (signal * samples);
  // if(tid < samples) printf("signal %d %d: %f\n", signal, tid, data[tid]);

  float2 ldata;
  ldata.x = data[0];
  ldata.y = -1;
  for (unsigned i = tid; i < samples; i += blockSize) {
    update(
      // comp,
      ldata,
      data,
      i);
  }
  // if (threadIdx.x == 0) printf("%f\n", ldata.x);
  __syncthreads();// wait till all threads are ready
  findUniversalInSharedMem(//<float2, blockSize>(
                           // comp,
    ldata,
    threadIdx.x);

  // last thread now holds the result
  if (tid == 0) {
    if (NULL != outVal) { outVal[signal] = ldata.x; }
    // if (NULL != outPos) { outPos[signal] = ldata.y; }
  }
}

template<typename T>
__global__ void findMaxRect(T *in,
  umpalumpa::data::Size inSize,
  T *outVal,
  T *outPos,
  unsigned offsetX,
  unsigned offsetY,
  unsigned rectWidth,
  unsigned rectHeight)
{
  unsigned signal = blockIdx.x;

  const float *data = in + (signal * inSize.single);

  float2 ldata;
  ldata.x = data[offsetY * inSize.x + offsetX];
  ldata.y = -1;
  for (unsigned tIdy = offsetY + threadIdx.y; tIdy < offsetY + rectHeight; tIdy += blockSizeY) {
    for (unsigned tIdx = offsetX + threadIdx.x; tIdx < offsetX + rectWidth; tIdx += blockSizeX) {
      update(ldata, data, tIdy * inSize.x + tIdx);
    }
  }

  __syncthreads();// wait till all threads are ready
  findUniversalInSharedMem(//<float2, blockSize>(
                           // comp,
    ldata,
    threadIdx.y * blockSizeX + threadIdx.x);

  // last thread now holds the result
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (NULL != outVal) { outVal[signal] = ldata.x; }
    if (NULL != outPos) { outPos[signal] = ldata.y; }
  }
}

// template <typename T, unsigned blockSize, typename C>
// __global__
// void findUniversal(
//         const C &comp,
//         T startVal,
//         const T * __restrict__ in,
//         float * __restrict__ outPos,
//         T * __restrict__ outVal,
//         unsigned samples)
// {
//     if (std::is_same<T, float> ::value) {
//         findMax1D<float, float2, blockSize> (
//                 comp,
//                 startVal,
//                 (float*)in,
//                 outPos,
//                 (float*)outVal,
//                 samples);
//     } else if (std::is_same<T, double> ::value) {
//         findMax1D<double, double2, blockSize>(
//                 comp,
//                 startVal,
//                 (double*)in,
//                 outPos,
//                 (double*)outVal,
//                 samples);
//     }
// }
