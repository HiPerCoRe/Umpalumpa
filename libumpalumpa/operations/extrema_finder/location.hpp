#pragma once

namespace umpalumpa::extrema_finder {

enum class Location {
  /**
   * FIXME this will probably have to be some abstract class, and then specific implementations will
   *hold additional info (i.e. distance from center etc)
   * FIXME implement other types, namely
   * RectCenter // i.e. in the rectangular area around the center
   * Window // general rectangular area
   * kAroundCenter,  // i.e. within radius from the center of the data
   *
   **/
  kEntire,// i.e. check entire data
  kWindow,// i.e. check rectangular area//TODO needs to know in what area to look (will be solved
          // with change to class instead of enum)
  kRectCenter,// i.e. check rectangular area around the center
};
}// namespace umpalumpa::extrema_finder
