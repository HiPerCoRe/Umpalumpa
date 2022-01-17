
#ifndef MILLISECONDS
#define MILLISECONDS 1000u// sleep for a second
#endif

__global__ void waitingKernel() { __nanosleep(1000000u * MILLISECONDS); }
