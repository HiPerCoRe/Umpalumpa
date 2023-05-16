#pragma once

#include <libumpalumpa/operations/basic_operation.hpp>
#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/logical_desriptor.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/operations/initialization/settings.hpp>

namespace umpalumpa::initialization {
template<typename T> struct OutputDataWrapper : public data::PayloadWrapper<T>
{
  OutputDataWrapper(std::tuple<T> &t) : data::PayloadWrapper<T>(t) {}
  OutputDataWrapper(T &data) : data::PayloadWrapper<T>(data) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
};

template<typename T> struct InputDataWrapper : public data::PayloadWrapper<T, T>
{
  InputDataWrapper(std::tuple<T, T> &t) : data::PayloadWrapper<T, T>(t) {}
  InputDataWrapper(T &data, T &value) : data::PayloadWrapper<T, T>(data, value) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
  const T &GetValue() const { return std::get<1>(this->payloads); };
};

class Abstract
  : public BasicOperation<OutputDataWrapper<data::Payload<data::LogicalDescriptor>>,
      InputDataWrapper<data::Payload<data::LogicalDescriptor>>,
      Settings>
{
public:
  bool IsValid(const OutputData &out, const InputData &in, const Settings &) const override
  {
    return in.IsValid() && out.IsValid() && in.GetData() == out.GetData()
           && in.GetValue().info.Elems() == 1;
  }
};
}// namespace umpalumpa::initialization