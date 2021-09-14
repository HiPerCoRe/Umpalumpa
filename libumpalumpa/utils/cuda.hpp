#pragma once

#include <libumpalumpa/system_includes/cuda_runtime.hpp>
#include <libumpalumpa/system_includes/cufftXt.hpp>
#include <cuda.h>

void(gpuErrchk)(cudaError_t code, const char *file, int line, bool abort = true);

void(gpuErrchk)(CUresult code, const char *file, int line, bool abort = true);

void(gpuErrchk)(cufftResult code, const char *file, int line, bool abort = true);

#define CudaErrchk(code)                   \
  {                                        \
    gpuErrchk((code), __FILE__, __LINE__); \
  }
