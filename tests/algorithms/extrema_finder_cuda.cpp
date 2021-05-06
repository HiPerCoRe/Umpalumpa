#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>
#include <gtest/gtest.h>

#include <cuda_runtime.h>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;

class SingleExtremaFinderCUDATest : public ::testing::Test
{
public:
  auto getSearcher() { return SingleExtremaFinderCUDA(0); }
  auto allocate(size_t bytes)
  {
    void *ptr;
    cudaMallocManaged(&ptr, bytes);
    return ptr;
  }

  void free(void *ptr) { cudaFree(ptr); }
};
#define NAME SingleExtremaFinderCUDATest
#include <tests/algorithms/extrema_finder_common.hpp>