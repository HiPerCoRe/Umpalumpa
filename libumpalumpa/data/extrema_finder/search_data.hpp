#ifndef LIBUMPALUMPA_DATA_EXTREMA_FINDER_SEARCH_DATA
#define LIBUMPALUMPA_DATA_EXTREMA_FINDER_SEARCH_DATA
#include <libumpalumpa/data/payload.hpp>

namespace umpalumpa {
namespace data {
namespace extrema_finder {
template <typename T>
class SearchData : public Payload<T> {
    using Payload<T>::Payload;
};
}  // namespace extrema_finder
}  // namespace data
}  // namespace umpalumpa
#endif /* LIBUMPALUMPA_DATA_EXTREMA_FINDER_SEARCH_DATA */
