#include <libumpalumpa/operations/reduction/cuda_kernels.cu>
#include <libumpalumpa/operations/reduction/cpu_kernels.hpp>
#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/utils/cuda.hpp>

namespace umpalumpa::fourier_reconstruction {

void InitCodeletCPU(void *buffers[], void *)
{
  // FIXME refactor once we change implementation of the buffer handling
  auto ptr = STARPU_VECTOR_GET_PTR(buffers[0]);
  auto size = STARPU_VECTOR_GET_NX(buffers[0]) * STARPU_VECTOR_GET_ELEMSIZE(buffers[0]);
  memset(reinterpret_cast<void *>(ptr), 0, size);
}

void InitCodeletCUDA(void *buffers[], void *)
{
  // FIXME refactor once we change implementation of the buffer handling
  auto ptr = STARPU_VECTOR_GET_PTR(buffers[0]);
  auto size = STARPU_VECTOR_GET_NX(buffers[0]) * STARPU_VECTOR_GET_ELEMSIZE(buffers[0]);
  CudaErrchk(
    cudaMemsetAsync(reinterpret_cast<void *>(ptr), 0, size, starpu_cuda_get_local_stream()));
}

void SumCodeletCPU(void *buffers[], void *)
{
  auto out = STARPU_VECTOR_GET_PTR(buffers[0]);
  auto in = STARPU_VECTOR_GET_PTR(buffers[1]);
  auto elems =
    STARPU_VECTOR_GET_NX(buffers[0]) * STARPU_VECTOR_GET_ELEMSIZE(buffers[0]) / sizeof(float);

  reduction::PiecewiseOp(reinterpret_cast<float *>(out),
    reinterpret_cast<float *>(in),
    data::Size(elems, 1, 1, 1),
    [](auto l, auto r) { return l + r; });
}

void SumCodeletCUDA(void *buffers[], void *)
{
  // FIXME refactor once we change implementation of the buffer handling
  auto out = STARPU_VECTOR_GET_PTR(buffers[0]);
  auto in = STARPU_VECTOR_GET_PTR(buffers[1]);
  auto elems =
    STARPU_VECTOR_GET_NX(buffers[0]) * STARPU_VECTOR_GET_ELEMSIZE(buffers[0]) / sizeof(float);
  dim3 blockSize(1024);
  dim3 gridSize(static_cast<unsigned int>(std::ceil(float(elems) / float(blockSize.x))));

  PiecewiseSum<<<gridSize, blockSize, 0, starpu_cuda_get_local_stream()>>>(
    reinterpret_cast<float *>(out), reinterpret_cast<float *>(in), data::Size(elems, 1, 1, 1));
}

}// namespace umpalumpa::fourier_reconstruction
