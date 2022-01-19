
#ifndef MILLISECONDS
#define MILLISECONDS 1000u// sleep for a second
#endif

#if __CUDA_ARCH__ >= 700
__global__ void waitingKernel() { __nanosleep(1000000u * MILLISECONDS); }
#else
// This version uses just very rough estimation of milliseconds
// We don't query the GPU's frequency dynamically, we chose the number 2 GHz

// Inspired by:
// https://stackoverflow.com/questions/11217117/equivalent-of-usleep-in-cuda-kernel
__device__ clock_t globalVar;

__global__ void waitingKernel()
{
  long long start = clock64();
  long long now;
  for (;;) {
    now = clock64();
    long long cycles = now - start;// ignore overflow... should not happen
    long long elapsedMiliseconds = (cycles / 2000000000.0f) * 1000.0f;
    if (elapsedMiliseconds >= MILLISECONDS) { break; }
  }
  // Makes sure that compiler doesn't optimize the loop away
  globalVar = now;
}
#endif
