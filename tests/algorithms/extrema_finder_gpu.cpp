#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_gpu.hpp>
#include <gtest/gtest.h>

#include <cuda_runtime.h>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;

class SingleExtremaFinderGPUTest : public ::testing::Test
{
public:
  auto getSearcher() { return SingleExtremaFinderGPU(0); }
  auto allocate(size_t bytes)
  {
    void *ptr;
    cudaMallocManaged(&ptr, bytes);
    return ptr;
  }

  void free(void *ptr) { cudaFree(ptr); }
};
#define NAME SingleExtremaFinderGPUTest
#include <tests/algorithms/extrema_finder_common.hpp>