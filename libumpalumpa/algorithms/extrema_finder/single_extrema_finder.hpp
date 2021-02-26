#ifndef LIBUMPALUMPA_ALGORITHMS_EXTREMA_FINDER_SINGLE_EXTREMA_FINDER
#define LIBUMPALUMPA_ALGORITHMS_EXTREMA_FINDER_SINGLE_EXTREMA_FINDER

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>

namespace umpalumpa {
namespace extrema_finder {
template <typename T>
class SingleExtremaFinder : public AExtremaFinder<T> {
   public:
    bool execute(const ResultData<T> &out, const SearchData<T> &in, const Settings &settings, bool dryRun) override;
};
}  // namespace extrema_finder
}  // namespace umpalumpa

#endif /* LIBUMPALUMPA_ALGORITHMS_EXTREMA_FINDER_SINGLE_EXTREMA_FINDER */
