#pragma once

#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include <libumpalumpa/algorithms/extrema_finder/settings.hpp>
#include <libumpalumpa/data/logical_desriptor.hpp>

namespace umpalumpa::extrema_finder {

template<typename T> struct OutputDataWrapper : public data::PayloadWrapper<T, T>
{
  OutputDataWrapper(std::tuple<T, T> &t) : data::PayloadWrapper<T, T>(t) {}
  OutputDataWrapper(T &vals, T &locs) : data::PayloadWrapper<T, T>(vals, locs) {}
  const T &GetValues() const { return std::get<0>(this->payloads); };
  const T &GetLocations() const { return std::get<1>(this->payloads); };
  typedef T PayloadType;
};

template<typename T> struct InputDataWrapper : public data::PayloadWrapper<T>
{
  InputDataWrapper(std::tuple<T> &t) : data::PayloadWrapper<T>(t) {}
  InputDataWrapper(T &data) : data::PayloadWrapper<T>(data) {}
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

    if (s.GetResult() == Result::kValue) {
      // is the type correct?
      result = result && (in.GetData().dataInfo.GetType() == out.GetValues().dataInfo.GetType());
      // we need to have enough space for results
      result = result && (in.GetData().info.GetSize().n == out.GetValues().info.GetSize().n);
      // output should be N 1D GetValues()
      result = result && (out.GetValues().info.GetSize().total == out.GetValues().info.GetSize().n);
    }
    if (s.GetResult() == Result::kLocation) {
      const auto &p = out.GetLocations();
      // is the type correct?
      result = result && (data::DataType::kFloat == p.dataInfo.GetType());
      // we need to have enough space for results
      result = result && (in.GetData().info.GetSize().n == p.info.GetSize().n);
      result = result && (data::Dimensionality::k1Dim == p.info.GetSize().GetDim());
      // results are stored in the DIM numbers in the x axis
      result = result && (in.GetData().info.GetSize().GetDimAsNumber() == p.info.GetSize().x);
    }
    return result;
  }
};
}// namespace umpalumpa::extrema_finder