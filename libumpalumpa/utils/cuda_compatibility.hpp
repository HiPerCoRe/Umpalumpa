#pragma once

#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

#ifdef __CUDACC__
#define CUDA_H __host__
#else
#define CUDA_H
#endif