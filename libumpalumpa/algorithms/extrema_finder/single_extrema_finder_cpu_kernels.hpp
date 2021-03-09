#ifndef LIBUMPALUMPA_ALGORITHMS_EXTREMA_FINDER_SINGLE_EXTREMA_FINDER_CPU_KERNELS
#define LIBUMPALUMPA_ALGORITHMS_EXTREMA_FINDER_SINGLE_EXTREMA_FINDER_CPU_KERNELS

#include <libumpalumpa/data/size.hpp>

namespace umpalumpa {
namespace extrema_finder {

template <typename T, typename C>
void FindSingleExtrema1D(T *vals, T *data, const umpalumpa::data::Size &size, const C &comp, const T &initVal) {
    // all checks are expected to be done by caller
    for (size_t n = 0; n < size.n; ++n) {
        const size_t offset = n * size.single;
        auto &extrema = const_cast<T &>(initVal);
        for (size_t i = 0; i < size.single; ++i) {
            auto &v = data[offset + i];
            if (comp(v, extrema)) {
                extrema = v;
            }
        }
        vals[n] = extrema;
    }
}
}  // namespace extrema_finder
}  // namespace umpalumpa

#endif /* LIBUMPALUMPA_ALGORITHMS_EXTREMA_FINDER_SINGLE_EXTREMA_FINDER_CPU_KERNELS */
