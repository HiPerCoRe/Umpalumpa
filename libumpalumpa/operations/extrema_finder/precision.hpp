#pragma once

namespace umpalumpa::extrema_finder {

/**
 * Precision of the found Location result
 **/
enum class Precision {
  kSingle,// i.e. we look for integer position of the extrema
  k3x3,// i.e. we look for the subpixel position of the extrema in the 3x3 window
};
}// namespace umpalumpa::extrema_finder
