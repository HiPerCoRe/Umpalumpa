#include <libumpalumpa/data/size.hpp>

template<typename T, bool center, bool isWithin>
__global__ void correlate2D(T *__restrict__ correlations,
  const T *__restrict__ in1,
  umpalumpa::data::Size in1Size,
  const T *__restrict__ in2,
  int in2N)
{
  // assign pixel to thread
#if TILE > 1
  int id = threadIdx.y * blockDim.x + threadIdx.x;
  int tidX = threadIdx.x % TILE + (id / (blockDim.y * TILE)) * TILE;
  int tidY = (id / TILE) % blockDim.y;
  int idx = blockIdx.x * blockDim.x + tidX;
  int idy = blockIdx.y * blockDim.y + tidY;
#else
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
#endif
  int centerCoef = 1 - 2 * ((idx + idy) & 1);// center FFT, input must be even

  // We assume that in1Size, in2Size are the same except for N
  if (idx >= in1Size.x || idy >= in1Size.y) return;
  size_t pixelIndex = idy * in1Size.x + idx;// index within single image

  int counter = 0;
  for (int i = 0; i <= in1Size.n; i++) {
    int tmpOffset = i * in1Size.single;
    T tmp = in1[tmpOffset + pixelIndex];
    for (int j = isWithin ? i + 1 : 0; j < in2N; j++) {
      int tmp2Offset = j * in1Size.single;
      T tmp2 = in2[tmp2Offset + pixelIndex];
      T res;
      res.x = (tmp.x * tmp2.x) + (tmp.y * tmp2.y);
      res.y = (tmp.y * tmp2.x) - (tmp.x * tmp2.y);
      if (center) {
        res.x *= centerCoef;
        res.y *= centerCoef;
      }
      correlations[counter * in1Size.single + pixelIndex] = res;
      counter++;
    }
  }
}

