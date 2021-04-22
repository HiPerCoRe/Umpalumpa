#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;

#define NAME ExtermaFinderCPU

auto getSearcher() {
  return SingleExtremaFinderCPU();
}

#include <tests/algorithms/extrema_finder_common.hpp>