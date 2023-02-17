#include <libumpalumpa/system_includes/cuda_runtime.hpp>

namespace umpalumpa::utils {

// Inspired by https://stackoverflow.com/a/47406068/5484355 and
// https://stackoverflow.com/a/70603147/5484355
template<typename T, auto fn> struct CudaErrchkDeleter
{
  constexpr void operator()(T *ptr) { CudaErrchk(fn(ptr)); }
};

auto CudaMallocAlloc = [](size_t bytes) {
  void *ptr;
  CudaErrchk(cudaMalloc(&ptr, bytes));
  return ptr;
};

template<typename T> using unique_ptr_cuda = std::unique_ptr<T, CudaErrchkDeleter<T, cudaFree>>;

template<typename T> auto make_unique_cuda(size_t bytes)
{
  return unique_ptr_cuda<T>(reinterpret_cast<T *>(CudaMallocAlloc(bytes)));
}
}// namespace umpalumpa::utils