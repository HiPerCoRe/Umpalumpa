#pragma once

#include <libumpalumpa/algorithms/extrema_finder/search_settings.hpp>
#include <libumpalumpa/data/extrema_finder/result_data.hpp>
#include <libumpalumpa/data/extrema_finder/search_data.hpp>

namespace umpalumpa {
namespace extrema_finder {
  using umpalumpa::extrema_finder::data::ResultData;
  using umpalumpa::extrema_finder::data::SearchData;
  class AExtremaFinder
  {
  public:
    virtual bool Init(const ResultData &out, const SearchData &in, const Settings &settings) = 0;
    virtual bool Execute(const ResultData &out, const SearchData &in, const Settings &settings) = 0;
    virtual void Cleanup(){};
    virtual ~AExtremaFinder() = default;

  protected:
    virtual bool
      IsValid(const ResultData &out, const SearchData &in, const Settings &settings) const
    {
      bool result = true;
      // is input valid?
      result = result && (settings.dryRun || (nullptr != in.data));
      result = result && in.IsValid();

      if (settings.result == SearchResult::kValue) {
        // is output valid?
        result = result && (nullptr != out.values);
        result = result && (out.values->IsValid());
        result = result && (settings.dryRun || (nullptr != out.values->data));
        // is the type correct?
        result = result && (in.dataInfo.type == out.values->dataInfo.type);
        // we need to have enough space for results
        result = result && (in.info.size.n == out.values->info.size.n);
        // output should be N 1D values
        result = result && (out.values->info.size.total == out.values->info.size.n);
      }
      return result;
    }
  };

}// namespace extrema_finder
}// namespace umpalumpa