#pragma once

#include <complex>
#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/operations/fourier_processing/settings.hpp>
#include <libumpalumpa/operations/basic_operation.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>
#include <libumpalumpa/data/logical_desriptor.hpp>

namespace umpalumpa::fourier_processing {

template<typename T, typename U> struct InputDataWrapper : public data::PayloadWrapper<T, U>
{
  InputDataWrapper(std::tuple<T, U> &t) : data::PayloadWrapper<T, U>(t) {}
  InputDataWrapper(T &data, U &filter) : data::PayloadWrapper<T, U>(data, filter) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
  const U &GetFilter() const { return std::get<1>(this->payloads); };
  typedef T DataType;
  typedef U FilterType;
};

template<typename T> struct OutputDataWrapper : public data::PayloadWrapper<T>
{
  OutputDataWrapper(std::tuple<T> &t) : data::PayloadWrapper<T>(t) {}
  OutputDataWrapper(T &data) : data::PayloadWrapper<T>(data) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
  typedef T DataType;
};

class AFP
  : public BasicOperation<OutputDataWrapper<data::Payload<data::FourierDescriptor>>,
      InputDataWrapper<data::Payload<data::FourierDescriptor>,
        data::Payload<data::LogicalDescriptor>>,
      Settings>
{

public:
  static bool IsFloat(const OutputData &out, const InputData &in)
  {
    return (in.GetData().dataInfo.GetType().Is<std::complex<float>>())
           && (in.GetFilter().dataInfo.GetType().Is<float>()
               || in.GetFilter().dataInfo.GetType().Is<void>())// no filter
           && (out.GetData().dataInfo.GetType().Is<std::complex<float>>());
  }

protected:
  virtual bool IsValid(const OutputData &out, const InputData &in, const Settings &) const
  {
    // FIXME add check that in case of the centering, data must be even (or odd, not sure which one)
    return out.IsValid() && in.IsValid();
  }
};
}// namespace umpalumpa::fourier_processing
