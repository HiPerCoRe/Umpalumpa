#include <starpu.h>

static __global__ void vector_mult_cuda(unsigned n, float* val, float factor) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) val[i] *= factor;
  if (i == 2) printf("hello from cuda\n");
}

extern "C" void scal_cuda_func(void* buffers[], void* _args) {
  float* factor = (float*)_args; /*length of the vector*/
  unsigned n =
      STARPU_VECTOR_GET_NX(buffers[0]); /*local copy of the vector pointer*/
  float* val = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
  unsigned threads_per_block = 64;
  unsigned nblocks = (n + threads_per_block - 1) / threads_per_block;
  vector_mult_cuda<<<nblocks, threads_per_block, 0,
                     starpu_cuda_get_local_stream()>>>(n, val, *factor);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
