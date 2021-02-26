namespace umpalumpa {
namespace extrema_finder {

enum class SearchType {
    /**
     * FIXME implement other types, namely
     * AbsMin, AbsMax,
     * Lowest, // i.e. the max negative number
     * Min, // i.e. the smallest finite number (positive number very close to zero)
     **/
    kMax,  // i.e. the max positive number
};
}  // namespace extrema_finder
}  // namespace umpalumpa