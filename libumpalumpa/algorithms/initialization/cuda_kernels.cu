#pragma once

#include <libumpalumpa/data/size.hpp>

template<typename T>
__global__ void Initialize(T *__restrict__ inOut, umpalumpa::data::Size size, T val)
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size.total) {
    inOut[i] = val;
  }
}