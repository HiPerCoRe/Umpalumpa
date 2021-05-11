#include <libumpalumpa/utils/cuda.hpp>
#include <cuda.h>
#include <cstdio>
#include <stdexcept>

void gpuErrchk(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) {
    char buffer[300];
    snprintf(buffer, 300, "GPUassert: %s %s %d", cudaGetErrorString(code), file, line);
    if (abort) throw std::runtime_error(std::string(buffer));
  }
}

void gpuErrchk(CUresult code, const char *file, int line, bool abort)
{
  if (code != CUDA_SUCCESS) {
    const char *name;
    cuGetErrorName(code, &name);
    const char *desc;
    cuGetErrorString(code, &desc);
    char buffer[600];
    snprintf(buffer, 600, "CUresult: %d %s %s\nFile %s line %d", code, name, desc, file, line);
    if (abort) throw std::runtime_error(std::string(buffer));
  }
}