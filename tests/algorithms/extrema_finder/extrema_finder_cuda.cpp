#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>
#include <gtest/gtest.h>

#include <cuda_runtime.h>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;

class SingleExtremaFinderCUDATest : public ::testing::Test
{
public:
  auto &GetSearcher() { return searcher; }
  auto Allocate(size_t bytes)
  {
    void *ptr;
    cudaMallocManaged(&ptr, bytes);
    return ptr;
  }

  void Free(void *ptr) { cudaFree(ptr); }

  void WaitTillDone() { searcher.Synchronize(); };

private:
  SingleExtremaFinderCUDA searcher = SingleExtremaFinderCUDA(0);
};
#define NAME SingleExtremaFinderCUDATest
#include <tests/algorithms/extrema_finder/extrema_finder_common.hpp>
