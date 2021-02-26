#ifndef LIBUMPALUMPA_DATA_EXTREMA_FINDER_RESULT_DATA
#define LIBUMPALUMPA_DATA_EXTREMA_FINDER_RESULT_DATA
#include <libumpalumpa/data/payload.hpp>

namespace umpalumpa {
namespace data {
namespace extrema_finder {
template <typename T>
class ResultData {
   public:
    ResultData(Payload<T> *values, Payload<T> *locations) : values(values), locations(locations) {}
    Payload<T> *const values;
    Payload<T> *const locations;
};
}  // namespace extrema_finder
}  // namespace data
}  // namespace umpalumpa
#endif /* LIBUMPALUMPA_DATA_EXTREMA_FINDER_RESULT_DATA */
