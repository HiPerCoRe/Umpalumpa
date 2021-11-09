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

static inline const char *_cudaGetErrorEnum(cufftResult error)
{
  switch (error) {
  case CUFFT_SUCCESS:
    return "CUFFT_SUCCESS";

  case CUFFT_INVALID_PLAN:
    return "CUFFT_INVALID_PLAN";

  case CUFFT_ALLOC_FAILED:
    return "CUFFT_ALLOC_FAILED";

  case CUFFT_INVALID_TYPE:
    return "CUFFT_INVALID_TYPE";

  case CUFFT_INVALID_VALUE:
    return "CUFFT_INVALID_VALUE";

  case CUFFT_INTERNAL_ERROR:
    return "CUFFT_INTERNAL_ERROR";

  case CUFFT_EXEC_FAILED:
    return "CUFFT_EXEC_FAILED";

  case CUFFT_SETUP_FAILED:
    return "CUFFT_SETUP_FAILED";

  case CUFFT_INVALID_SIZE:
    return "CUFFT_INVALID_SIZE";

  case CUFFT_UNALIGNED_DATA:
    return "CUFFT_UNALIGNED_DATA";

  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    return "CUFFT_INCOMPLETE_PARAMETER_LIST";

  case CUFFT_INVALID_DEVICE:
    return "CUFFT_INVALID_DEVICE";

  case CUFFT_PARSE_ERROR:
    return "CUFFT_PARSE_ERROR";

  case CUFFT_NO_WORKSPACE:
    return "CUFFT_NO_WORKSPACE";
#if CUDART_VERSION > 7050
  case CUFFT_NOT_IMPLEMENTED:
    return "CUFFT_NOT_IMPLEMENTED";

  case CUFFT_LICENSE_ERROR:
    return "CUFFT_LICENSE_ERROR";

  case CUFFT_NOT_SUPPORTED:
    return "CUFFT_NOT_SUPPORTED";
#endif
  }

  return "<unknown>";
}

void gpuErrchk(cufftResult code, const char *file, int line, bool abort)
{
  if (code != cufftResult::CUFFT_SUCCESS) {
    char buffer[300];
    snprintf(buffer, 300, "GPUassert: %s %s %d", _cudaGetErrorEnum(code), file, line);
    if (abort) throw std::runtime_error(std::string(buffer));
  }
}

bool IsGpuPointer(const void *p)
{
  cudaPointerAttributes attr;
  if (cudaPointerGetAttributes(&attr, p) == cudaErrorInvalidValue) {
    cudaGetLastError();// clear out the previous API error
    return false;
  }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
  return cudaMemoryTypeDevice == attr.type;
#else
  return cudaMemoryTypeDevice == attr.memoryType;
#endif
}