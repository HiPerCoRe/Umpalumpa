#include <starpu.h>

#define REAL float
#define REAL2 float2
#define REAL4 float4
#define CACHING_STRATEGY 1
#define GROUP_SIZE_Y 4
#define GROUP_SIZE_Z 4
#define PADD_AA 0
#define PADD_AB 0
#define PADD_C 0
#define DIRECT_WRITE 0
#define UNROLL_K 1
#define SIZE_A 16
#define SIZE_B 16
#define SIZE_C 16

static __global__ void gemm_batch_kernel(const REAL* A, const REAL* B, REAL* C, int n) {
      
    int matrix = blockIdx.x*GROUP_SIZE_Z + threadIdx.z;
    int matrixBatch = blockIdx.x*GROUP_SIZE_Z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

/* preload data */
 #if CACHING_STRATEGY == 1 or CACHING_STRATEGY == 2
    int preloadStartA = matrixBatch*SIZE_A*SIZE_B;
    int preloadStartB = matrixBatch*SIZE_C*SIZE_A;
    int myOffset = tz*(SIZE_C+PADD_C)*GROUP_SIZE_Y + ty*(SIZE_C+PADD_C) + tx;
    __shared__ REAL bufA[GROUP_SIZE_Z*(SIZE_A+PADD_AA)*(SIZE_B+PADD_AB)];
  #if CACHING_STRATEGY == 1 or GROUP_SIZE_Y > 1
    __shared__ REAL bufB[GROUP_SIZE_Z*SIZE_C*SIZE_A + PADD_C];
  #endif
  #if DIRECT_WRITE == 0
    __shared__ REAL bufC[GROUP_SIZE_Z*SIZE_C*SIZE_B];
  #endif
    for (int i = myOffset; i < SIZE_A*SIZE_B*GROUP_SIZE_Z; i+= (SIZE_C+PADD_C)*GROUP_SIZE_Y*GROUP_SIZE_Z) {
  #if PADD_AA == 0
        bufA[i] = A[preloadStartA + i];
  #else
        int padd = (i/SIZE_A)*PADD_AA;
        bufA[i+padd] = A[preloadStartA + i];
  #endif
    }
/*    #if PADD_AA > 0
    if (myOffset < SIZE_B)
        for (int i = 0; i < PADD_AA; i++)
            bufA[tz*(SIZE_A+PADD_AA)*SIZE_B + myOffset*(SIZE_A+PADD_AA) + i] = (REAL)0.0;
    #endif*/
  #if CACHING_STRATEGY == 1 or GROUP_SIZE_Y > 1
    for (int i = myOffset; i < SIZE_C*SIZE_A*GROUP_SIZE_Z; i+= (SIZE_C+PADD_C)*GROUP_SIZE_Y*GROUP_SIZE_Z) {
        bufB[i] = B[preloadStartB + i];
    }
  #endif
    __syncthreads();
 #endif
/* offsets into memory */
 #if CACHING_STRATEGY == 0
    int startA = matrix*SIZE_A*SIZE_B;
    int startB = matrix*SIZE_C*SIZE_A;
 #else
    int startA = tz*(SIZE_A+PADD_AA)*SIZE_B;
  #if CACHING_STRATEGY == 2 and GROUP_SIZE_Y == 1
    int startB = matrix*SIZE_C*SIZE_A;
  #else
    int startB = tz*SIZE_C*SIZE_A;
  #endif
 #endif
 #if DIRECT_WRITE == 0
    int startC = matrixBatch*SIZE_C*SIZE_B;
 #else
    int startC = matrix*SIZE_C*SIZE_B;
 #endif

/* compute multiplication */
 #if CACHING_STRATEGY < 2
  #if PADD_C > 0
    if (tx < SIZE_C) {
  #endif
    for (int i = ty; i < SIZE_B; i+= GROUP_SIZE_Y) {
        REAL tmp = (REAL)0.0;
        for (int k = 0; k < SIZE_A; k++)
  #if CACHING_STRATEGY == 0
            tmp += A[startA + i*SIZE_A + k] * B[startB + k*SIZE_C + tx];
  #endif
  #if CACHING_STRATEGY == 1
            tmp += bufA[startA + i*(SIZE_A+PADD_AA) + k] * bufB[startB + k*SIZE_C + tx];
  #endif
  #if DIRECT_WRITE == 0
        bufC[tz*SIZE_C*SIZE_B + i*SIZE_C + tx] = tmp;
  #else
        C[startC + i*SIZE_C + tx] = tmp;
  #endif
    }
  #if PADD_C > 0
    }
  #endif
 #else /* CACHING_STRATEGY == 2*/
    const int batch_base = (SIZE_B+PADD_AB)/GROUP_SIZE_Y;
    const int batch_peel = (SIZE_B+PADD_AB+GROUP_SIZE_Y-1)/GROUP_SIZE_Y;
    REAL tmp[batch_peel];
    for (int i = 0; i < batch_peel; i++)
        tmp[i] = REAL(0.0);
    int k;
  #if UNROLL_K == 1
    for (k = 0; k < SIZE_A-3; k+=4) {
   #if GROUP_SIZE_Y > 1
        REAL myB0 = bufB[startB + k*SIZE_C + tx];
        REAL myB1 = bufB[startB + (k+1)*SIZE_C + tx];
        REAL myB2 = bufB[startB + (k+2)*SIZE_C + tx];
        REAL myB3 = bufB[startB + (k+3)*SIZE_C + tx];
   #else
        REAL myB0 = B[startB + k*SIZE_C + tx];
        REAL myB1 = B[startB + (k+1)*SIZE_C + tx];
        REAL myB2 = B[startB + (k+2)*SIZE_C + tx];
        REAL myB3 = B[startB + (k+3)*SIZE_C + tx];
   #endif
        for (int i = 0; i < batch_base; i++) {
            tmp[i] += bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k] * myB0;
            tmp[i] += bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k+1] * myB1;
            tmp[i] += bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k+2] * myB2;
            tmp[i] += bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k+3] * myB3;
        }
        for (int i = batch_base; i < batch_peel; i++) {
            tmp[i] += (i*GROUP_SIZE_Y+ty < SIZE_B) ? bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k] * myB0 : (REAL)0.0;
            tmp[i] += (i*GROUP_SIZE_Y+ty < SIZE_B) ? bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k+1] * myB1 : (REAL)0.0;
            tmp[i] += (i*GROUP_SIZE_Y+ty < SIZE_B) ? bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k+2] * myB2 : (REAL)0.0;
            tmp[i] += (i*GROUP_SIZE_Y+ty < SIZE_B) ? bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k+3] * myB3 : (REAL)0.0;
        }
    }
    for (; k < SIZE_A-1; k+=2) {
   #if GROUP_SIZE_Y > 1
        REAL myB0 = bufB[startB + k*SIZE_C + tx];
        REAL myB1 = bufB[startB + (k+1)*SIZE_C + tx];
   #else
        REAL myB0 = B[startB + k*SIZE_C + tx];
        REAL myB1 = B[startB + (k+1)*SIZE_C + tx];
   #endif
        for (int i = 0; i < batch_base; i++) {
            tmp[i] += bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k] * myB0;
            tmp[i] += bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k+1] * myB1;
        }
        for (int i = batch_base; i < batch_peel; i++) {
            tmp[i] += (i*GROUP_SIZE_Y+ty < SIZE_B) ? bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k] * myB0 : (REAL)0.0;
            tmp[i] += (i*GROUP_SIZE_Y+ty < SIZE_B) ? bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k+1] * myB1 : (REAL)0.0;
        }
    }
  #else
  k = 0;
  #endif
    for (; k < SIZE_A; k++) {
  #if GROUP_SIZE_Y > 1
        REAL myB = bufB[startB + k*SIZE_C + tx];
  #else
        REAL myB = B[startB + k*SIZE_C + tx];
  #endif
        for (int i = 0; i < batch_base; i++) {
            tmp[i] += bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k] * myB;
        }
        for (int i = batch_base; i < batch_peel; i++) {
            tmp[i] += (i*GROUP_SIZE_Y+ty < SIZE_B) ? bufA[startA + (i*GROUP_SIZE_Y+ty)*(SIZE_A+PADD_AA) + k] * myB : (REAL)0.0;
        }
    }
  #if PADD_C > 0
    if (tx < SIZE_C) {
  #endif
    for (int i = 0; i < batch_peel; i++) {
        int index = i*GROUP_SIZE_Y+ty;
        if (index < SIZE_B)
  #if DIRECT_WRITE == 0
            bufC[tz*SIZE_C*SIZE_B + index*SIZE_C + tx] = tmp[i];
  #else
            C[startC + index*SIZE_C + tx] = tmp[i];
  #endif
    }
  #if PADD_C > 0
    }
  #endif
 #endif
 #if DIRECT_WRITE == 0
    __syncthreads();
    for (int i = myOffset; i < SIZE_C*SIZE_B*GROUP_SIZE_Z; i+= SIZE_C*GROUP_SIZE_Y*GROUP_SIZE_Z) {
        C[startC + i] = bufC[i];
    }
 #endif
}



extern "C" void gemm_batch(void *buffers[], void *_args){

    //processing the data from buffers
    const float* A = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
    const float* B = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
    float* C = (float*)STARPU_VECTOR_GET_PTR(buffers[2]);
   
    int* arguments = (int*)_args;
    int n = arguments[3];

    unsigned threads_per_block = 64;
    unsigned nblocks = (n + threads_per_block-1) / threads_per_block;

    gemm_batch_kernel<<<nblocks,threads_per_block,0,starpu_cuda_get_local_stream()>>>(A, B, C, n);
}

