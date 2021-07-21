#pragma once

namespace umpalumpa {
namespace extrema_finder {

  enum class SearchResult { // FIXME check if we can do bitmask from enum class values https://github.com/HiPerCoRe/KTT/blob/master/Source/Utility/BitfieldEnum.h
    /**
     * FIXME implement other options, namely
     * kPosition (float)
     * kBoth (kPosition and kValue)
     **/
    kValue,// i.e. retrieve only the values
  };

}// namespace extrema_finder
}// namespace umpalumpa