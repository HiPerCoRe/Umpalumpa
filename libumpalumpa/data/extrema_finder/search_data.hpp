#pragma once

#include <libumpalumpa/data/payload.hpp>

namespace umpalumpa {
namespace extrema_finder {
  namespace data {
    class SearchData : public umpalumpa::data::Payload
    {
      using Payload::Payload;
    };
  }// namespace data
}// namespace extrema_finder
}// namespace umpalumpa
