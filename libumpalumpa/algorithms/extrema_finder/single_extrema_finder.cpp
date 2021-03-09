#include <functional>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu_kernels.hpp>
#include <limits>

namespace umpalumpa {
namespace extrema_finder {

  template<typename T>
  bool SingleExtremaFinder<T>::execute(const ResultData<T> &out,
    const SearchData<T> &in,
    const Settings &settings,
    bool dryRun)
  {
    // FIXME implement checks
    if (dryRun) { return true; }
    if (settings.size.dim == data::Dimensionality::k1Dim) {
      // conditions: no padding, search for max
      FindSingleExtrema1D(out.values->data, in.data, in.info.size, std::greater<T>(), std::numeric_limits<T>::lowest());
      return true;
    }
    return false;
  }

  template class SingleExtremaFinder<float>;
}// namespace extrema_finder
}// namespace umpalumpa
