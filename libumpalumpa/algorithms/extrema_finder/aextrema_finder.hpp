#pragma once

#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include <libumpalumpa/algorithms/extrema_finder/search_settings.hpp>
#include <libumpalumpa/data/logical_desriptor.hpp>

namespace umpalumpa::extrema_finder {

template<typename T> struct OutputDataWrapper : public data::PayloadWrapper<T, T>
{
  OutputDataWrapper(std::tuple<T, T> &&t) : data::PayloadWrapper<T, T>(std::move(t)) {}
  OutputDataWrapper(T vals, T locs) : data::PayloadWrapper<T, T>(std::move(vals), std::move(locs))
  {}
  const T &GetValues() const { return std::get<0>(this->payloads); };
  const T &GetLocations() const { return std::get<1>(this->payloads); };
  static OutputDataWrapper ValuesOnly(T vals)
  {
    return OutputDataWrapper(std::move(vals), T(vals.info, "Default Locations"));
  }

  static OutputDataWrapper LocationsOnly(T locs)
  {
    return OutputDataWrapper(T(locs.info, "Default Values"), std::move(locs));
  }
  typedef T PayloadType;
};

template<typename T> struct InputDataWrapper : public data::PayloadWrapper<T>
{
  InputDataWrapper(std::tuple<T> &&t) : data::PayloadWrapper<T>(std::move(t)) {}
  InputDataWrapper(T data) : data::PayloadWrapper<T>(std::move(data)) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
  typedef T PayloadType;
};

class AExtremaFinder
  : public BasicAlgorithm<OutputDataWrapper<data::Payload<data::LogicalDescriptor>>,
      InputDataWrapper<data::Payload<data::LogicalDescriptor>>,
      Settings>
{
protected:
  virtual bool IsValid(const OutputData &out, const InputData &in, const Settings &s) const
  {
    // is input valid?
    bool result = in.GetData().IsValid();

    if (s.GetResult() == SearchResult::kValue) {
      // is output valid?
      result = result && (out.GetValues().IsValid());
      // is the type correct?
      result = result && (in.GetData().dataInfo.GetType() == out.GetValues().dataInfo.GetType());
      // we need to have enough space for results
      result = result && (in.GetData().info.GetSize().n == out.GetValues().info.GetSize().n);
      // output should be N 1D GetValues()
      result = result && (out.GetValues().info.GetSize().total == out.GetValues().info.GetSize().n);
    }
    return result;
  }
};
}// namespace umpalumpa::extrema_finder