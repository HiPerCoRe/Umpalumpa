namespace umpalumpa {
namespace extrema_finder {

enum class SearchLocation {
    /**
     * FIXME this will probably have to be some abstract class, and then specific implementations will hold additional
     * info (i.e. distance from center etc)
     * FIXME implement other types, namely
     * RectCenter // i.e. in the rectangular area around the center
     * Window // general rectangular area
     * kAroundCenter,  // i.e. within radius from the center of the data
     *
     **/
    kEntire,  // i.e. check entire data
};

}  // namespace extrema_finder
}  // namespace umpalumpa