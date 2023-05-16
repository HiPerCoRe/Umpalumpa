#pragma once

#include <libumpalumpa/data/size.hpp>

__global__ void
  PiecewiseSum(float *__restrict__ out, float *__restrict__ in, umpalumpa::data::Size size)
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size.total / 4) {
    auto *t = reinterpret_cast<float4 *>(out);
    auto *s = reinterpret_cast<float4 *>(in);
    t[i].x += s[i].x;
    t[i].y += s[i].y;
    t[i].z += s[i].z;
    t[i].w += s[i].w;
  }
  if (i == 0) {
    for (i = (size.total / 4) * 4; i < size.total; ++i) { out[i] += in[i]; }
  }
}