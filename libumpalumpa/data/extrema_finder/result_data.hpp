#pragma once

#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/data//extrema_finder/search_data.hpp>

namespace umpalumpa {
namespace extrema_finder {
  namespace data {
    class ResultData
    {
    public:
      ResultData(umpalumpa::data::Payload *vals, umpalumpa::data::Payload *locs) : values(vals), locations(locs) {}
      umpalumpa::data::Payload *const values;
      umpalumpa::data::Payload *const locations;
    };
  }// namespace data
}// namespace extrema_finder
}// namespace umpalumpa
