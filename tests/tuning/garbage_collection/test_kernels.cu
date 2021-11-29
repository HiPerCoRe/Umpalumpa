__global__ void TestKernel1(int *ptr)
{
  if (ptr != nullptr) { *ptr = 1; }
}

__global__ void TestKernel2(int *ptr)
{
  if (ptr != nullptr) { *ptr = 2; }
}

__global__ void TestKernel3(int *ptr)
{
  if (ptr != nullptr) { *ptr = 3; }
}
