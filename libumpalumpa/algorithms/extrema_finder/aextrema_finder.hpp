#pragma once

#include <optional>
#include <libumpalumpa/algorithms/extrema_finder/search_settings.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/data/logical_desriptor.hpp>

namespace umpalumpa {
namespace extrema_finder {
  class AExtremaFinder
  {
  protected:
    template<typename T> struct ResultDataWrapper
    {
      ResultDataWrapper(const std::optional<T> &vals, const std::optional<T> &locs)
        : values(vals), locations(locs)
      {}
      const std::optional<T> values;
      const std::optional<T> locations;
      typedef T type;
    };

    template<typename T> struct SearchDataWrapper
    {
      SearchDataWrapper(const T &d) : data(d) {}
      const T data;
      typedef T type;
    };

  public:
    using ResultData =
      ResultDataWrapper<umpalumpa::data::Payload<umpalumpa::data::LogicalDescriptor>>;
    using SearchData =
      SearchDataWrapper<umpalumpa::data::Payload<umpalumpa::data::LogicalDescriptor>>;
    virtual bool Init(const ResultData &out, const SearchData &in, const Settings &settings) = 0;
    virtual bool Execute(const ResultData &out, const SearchData &in, const Settings &settings) = 0;
    virtual void Cleanup(){};
    virtual void Synchronize() = 0;

    virtual ~AExtremaFinder() = default;

  protected:
    virtual bool
      IsValid(const ResultData &out, const SearchData &in, const Settings &settings) const
    {
      // is input valid?
      bool result = in.data.IsValid() && !in.data.IsEmpty();

      if (settings.result == SearchResult::kValue) {
        // is output valid?
        result = result && out.values;
        result = result && (out.values->IsValid());
        result = result && (nullptr != out.values->data);
        // is the type correct?
        result = result && (in.data.dataInfo.type == out.values->dataInfo.type);
        // we need to have enough space for results
        result = result && (in.data.info.size.n == out.values->info.size.n);
        // output should be N 1D values
        result = result && (out.values->info.size.total == out.values->info.size.n);
      }
      return result;
    }
  };

}// namespace extrema_finder
}// namespace umpalumpa