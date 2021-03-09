#include <libumpalumpa/algorithms/extrema_finder/search_location.hpp>
#include <libumpalumpa/algorithms/extrema_finder/search_result.hpp>
#include <libumpalumpa/algorithms/extrema_finder/search_type.hpp>
#include <libumpalumpa/data/size.hpp>

namespace umpalumpa {
namespace extrema_finder {

class Settings {
   public:
    explicit Settings(const SearchType &t, const SearchLocation &l, const SearchResult &r, const data::Size &s,
                      size_t batchSize)
        : type(t), location(l), size(s), batch(batchSize), result(r) {}
    const SearchType type;
    const SearchLocation location;
    const data::Size size;
    const size_t batch;
    const SearchResult result;

    bool IsValid() const { return size.IsValid() and (batch <= size.n); }
};

}  // namespace extrema_finder
}  // namespace umpalumpa