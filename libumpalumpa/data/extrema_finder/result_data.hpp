#pragma once

#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/data/logical_desriptor.hpp>

namespace umpalumpa {
namespace extrema_finder {
  namespace data {
    class ResultData
    {
    public:
      ResultData(umpalumpa::data::Payload<umpalumpa::data::LogicalDescriptor> *vals,
        umpalumpa::data::Payload<umpalumpa::data::LogicalDescriptor> *locs)
        : values(vals), locations(locs)
      {}
      umpalumpa::data::Payload<umpalumpa::data::LogicalDescriptor> *const values;
      umpalumpa::data::Payload<umpalumpa::data::LogicalDescriptor> *const locations;
    };
  }// namespace data
}// namespace extrema_finder
}// namespace umpalumpa
