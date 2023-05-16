#pragma once

#include <libumpalumpa/operations/basic_operation.hpp>
#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/logical_desriptor.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/operations/reduction/settings.hpp>

namespace umpalumpa::reduction {
template<typename T> struct OutputDataWrapper : public data::PayloadWrapper<T>
{
  OutputDataWrapper(std::tuple<T> &t) : data::PayloadWrapper<T>(t) {}
  OutputDataWrapper(T &data) : data::PayloadWrapper<T>(data) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
};

template<typename T> struct InputDataWrapper : public data::PayloadWrapper<T>
{
  InputDataWrapper(std::tuple<T> &t) : data::PayloadWrapper<T>(t) {}
  InputDataWrapper(T &data) : data::PayloadWrapper<T>(data) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
};

class Abstract
  : public BasicOperation<OutputDataWrapper<data::Payload<data::LogicalDescriptor>>,
      InputDataWrapper<data::Payload<data::LogicalDescriptor>>,
      Settings>
{
public:
  bool IsValid(const OutputData &out, const InputData &in, const Settings &) const override
  {
    return in.IsValid() && out.IsValid()
           && in.GetData().dataInfo.GetType() == out.GetData().dataInfo.GetType()
           && in.GetData().info.Elems() == out.GetData().info.Elems();
  }
};
}// namespace umpalumpa::reduction