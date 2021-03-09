#include <libumpalumpa/algorithms/extrema_finder/search_settings.hpp>
#include <libumpalumpa/data/extrema_finder/result_data.hpp>
#include <libumpalumpa/data/extrema_finder/search_data.hpp>

namespace umpalumpa {
namespace extrema_finder {
using umpalumpa::data::extrema_finder::ResultData;
using umpalumpa::data::extrema_finder::SearchData;
template <typename T>
class AExtremaFinder {
   public:
    virtual bool execute(const ResultData<T> &out, const SearchData<T> &in, const Settings &settings, bool dryRun) = 0;
    virtual ~AExtremaFinder() = default;
};

}  // namespace extrema_finder
}  // namespace umpalumpa