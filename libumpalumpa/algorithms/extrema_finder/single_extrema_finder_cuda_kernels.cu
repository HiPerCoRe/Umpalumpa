#pragma once
#include <libumpalumpa/data/size.hpp>

#ifndef blockSize
#define blockSize (blockSizeX * blockSizeY)
#endif

template<typename T, typename T2, typename C>
__device__ void update(
  const C &comp,
  T2 &orig,
  const T *__restrict data,
  unsigned index)
{
  T tmp = data[index];
  if (comp(tmp, index, orig.x, orig.y)) {
    orig.x = tmp;
    orig.y = (T)index;
  }
}

template<typename T, typename C>
__device__ T update(
  const C &comp,
  T &orig,
  T &cand)
{
  if (comp(cand.x, cand.y, orig.x, orig.y)) {
    orig.x = cand.x;
    orig.y = cand.y;
  }
  return orig;
}

template<typename T, typename C>
__device__ void findUniversalInSharedMem(
  const C &comp,
  T &ldata,
  unsigned int tid)
{
  // we have read all data, one of the thread knows the result
  __shared__ T sdata[blockSize];
  sdata[tid] = ldata;
  __syncthreads();// wait till all threads store their data
  // reduce
#pragma unroll
  for (auto counter = blockSize / 2; counter >= 32; counter /= 2) {
    if (tid < counter) {
      sdata[tid] = update(
        comp,
        ldata,
        sdata[tid + counter]);
    }
    __syncthreads();
  }
  // manually unwrap last warp for better performance
  // many of these blocks will be optimized out by compiler based on template
  if ((blockSize >= 32) && (tid < 16)) {
    sdata[tid] = update(
      comp,
      ldata,
      sdata[tid + 16]);
  }
  __syncthreads();
  if ((blockSize >= 16) && (tid < 8)) {
    sdata[tid] = update(
      comp,
      ldata,
      sdata[tid + 8]);
  }
  __syncthreads();
  if ((blockSize >= 8) && (tid < 4)) {
    sdata[tid] = update(
      comp,
      ldata,
      sdata[tid + 4]);
  }
  __syncthreads();
  if ((blockSize >= 4) && (tid < 2)) {
    sdata[tid] = update(
      comp,
      ldata,
      sdata[tid + 2]);
  }
  __syncthreads();
  if ((blockSize >= 2) && (tid < 1)) {
    sdata[tid] = update(
      comp,
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
extern "C" __global__ void findMax(
  // const C &comp,
  // T startVal,
  float * __restrict__ in,
  float * __restrict__ outVal,
  float * __restrict__ outPos,
  umpalumpa::data::Size size)
{
  // return true IFF first value is bigger than the second value, or they are the same and
  // the positio of the first value is lower -> returns the biggers value at the lowest position
  auto comp = [] (float l, unsigned li, float r, unsigned ri) { return l > r || (l == r && li < ri); };
  // one block processes one signal
  // map each thread to some sample of the signal
  // blockSize == noOfThreads
  auto tid = threadIdx.x;
  auto signal = blockIdx.x;
  
  // load data from global memory
  if (tid >= size.single) return;
  auto samples = size.single;
  const float *data = in + (signal * samples);

  float2 ldata;
  ldata.x = data[tid];
  ldata.y = tid;
  for (auto i = tid + blockSize; i < samples; i += blockSize) {
    update(
      comp,
      ldata,
      data,
      i);
  }
  __syncthreads();// wait till all threads are ready
  findUniversalInSharedMem(//<float2, blockSize>(
    comp,
    ldata,
    tid);

  // last thread now holds the result
  if (tid == 0) {
    if (nullptr != outVal) { outVal[signal] = ldata.x; }
    if (nullptr != outPos) { 
      const auto dim = size.GetDimAsNumber();
      const unsigned location = ldata.y;
      auto *dest = outPos + signal * dim;
      if (1 == dim) {
        dest[0] = static_cast<float>(location);
      } else if (2 == dim) {
        auto y = location / size.x;
        auto x = location % size.x;
        dest[0] = x;
        dest[1] = y;
      } else if (3 == dim) {
        size_t z = location / (size.x * size.y);
        size_t tmp = location % (size.x * size.y);
        size_t y = tmp / size.x;
        size_t x = tmp % size.x;
        dest[0] = static_cast<float>(x);
        dest[1] = static_cast<float>(y);
        dest[2] = static_cast<float>(z);
      } else {
        dest[signal] = nanf("");
      }
    }
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

  // return true IFF first value is bigger than the second value, or they are the same and
  // the positio of the first value is lower -> returns the biggers value at the lowest position
  auto comp = [] (float l, unsigned li, float r, unsigned ri) { return l > r || (l == r && li < ri); };

  float2 ldata;
  ldata.x = data[offsetY * inSize.x + offsetX];
  ldata.y = -1;
  for (unsigned tIdy = offsetY + threadIdx.y; tIdy < offsetY + rectHeight; tIdy += blockSizeY) {
    for (unsigned tIdx = offsetX + threadIdx.x; tIdx < offsetX + rectWidth; tIdx += blockSizeX) {
      update(comp, ldata, data, tIdy * inSize.x + tIdx);
    }
  }

  __syncthreads();// wait till all threads are ready
  findUniversalInSharedMem(//<float2, blockSize>(
    comp,
    ldata,
    threadIdx.y * blockSizeX + threadIdx.x);

  // last thread now holds the result
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (nullptr != outVal) { outVal[signal] = ldata.x; }
    if (nullptr != outPos) { outPos[signal] = ldata.y; }
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
