#pragma once

#include <libumpalumpa/algorithms/extrema_finder/search_settings.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/data/logical_desriptor.hpp>

namespace umpalumpa::extrema_finder {
class AExtremaFinder
{
protected:
  template<typename T> struct ResultDataWrapper
  {
    ResultDataWrapper(T &&vals, T &&locs) : values(std::move(vals)), locations(std::move(locs)) {}
    const T values;
    const T locations;
    typedef T type;
  };

  template<typename T> struct SearchDataWrapper
  {
    SearchDataWrapper(T &&d) : data(std::move(d)) {}
    const T data;
    typedef T type;
  };

public:
  struct OutputData final
    : public ResultDataWrapper<data::Payload<umpalumpa::data::LogicalDescriptor>>
  {
    OutputData(type &&vals, type &&locs) : ResultDataWrapper(std::move(vals), std::move(locs)) {}

    static OutputData ValuesOnly(type vals)
    {
      return OutputData(std::move(vals), type(vals.info, "Default Locations"));
    }

    static OutputData LocationsOnly(type locs)
    {
      return OutputData(type(locs.info, "Default Values"), std::move(locs));
    }
  };
  using InputData = SearchDataWrapper<data::Payload<data::LogicalDescriptor>>;
  virtual bool Init(const OutputData &out,
    const InputData &in,
    const Settings &settings) = 0;// FIXME add nodiscard?
  virtual bool Execute(const OutputData &out,
    const InputData &in,
    const Settings &settings) = 0;// setting is probably useless here, save it in Init()
  virtual void Cleanup(){};
  virtual void Synchronize() = 0;

  virtual ~AExtremaFinder() = default;

protected:
  virtual bool IsValid(const OutputData &out,
    const InputData &in,
    const Settings &settings) const// move to cpp
  {
    // is input valid?
    bool result = in.data.IsValid() && !in.data.IsEmpty();

    if (settings.result == SearchResult::kValue) {
      // is output valid?
      result = result && (out.values.IsValid());
      result = result && !out.values.IsEmpty();
      // is the type correct?
      result = result && (in.data.dataInfo.type == out.values.dataInfo.type);
      // we need to have enough space for results
      result = result && (in.data.info.GetSize().n == out.values.info.GetSize().n);
      // output should be N 1D values
      result = result && (out.values.info.GetSize().total == out.values.info.GetSize().n);
    }
    return result;
  }
};
}// namespace umpalumpa::extrema_finder