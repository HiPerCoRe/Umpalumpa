#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_gpu.hpp>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;

#define NAME ExtermaFinderGPU

auto getSearcher() { return SingleExtremaFinderGPU(); }

#include <tests/algorithms/extrema_finder_common.hpp>