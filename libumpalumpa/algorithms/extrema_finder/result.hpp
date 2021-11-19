#pragma once

namespace umpalumpa::extrema_finder {

enum class Result {// FIXME check if we can do bitmask from enum class values
                   // https://github.com/HiPerCoRe/KTT/blob/master/Source/Utility/BitfieldEnum.h
  /**
   * FIXME implement other options, namely
   * kPosition (float)
   * kBoth (kPosition and kValue)
   **/
  kValue,// i.e. retrieve only the values
  kLocation,// i.e. retieve only the position
};
}// namespace umpalumpa::extrema_finder
