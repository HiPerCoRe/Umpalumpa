#pragma once

#include <complex>
#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/operations/basic_operation.hpp>
#include <libumpalumpa/operations/correlation/settings.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>

namespace umpalumpa::correlation {

template<typename T = data::Payload<data::FourierDescriptor>>
struct InputDataWrapper : public data::PayloadWrapper<T, T>
{
  InputDataWrapper(std::tuple<T, T> &t) : data::PayloadWrapper<T, T>(t) {}
  InputDataWrapper(T &d1, T &d2) : data::PayloadWrapper<T, T>(d1, d2) {}
  const T &GetData1() const { return std::get<0>(this->payloads); };
  const T &GetData2() const { return std::get<1>(this->payloads); };
  typedef T PayloadType;
};

template<typename T = data::Payload<data::FourierDescriptor>>
struct OutputDataWrapper : public data::PayloadWrapper<T>
{
  OutputDataWrapper(std::tuple<T> &t) : data::PayloadWrapper<T>(t) {}
  OutputDataWrapper(T &correlations) : data::PayloadWrapper<T>(correlations) {}
  const T &GetCorrelations() const { return std::get<0>(this->payloads); };
  typedef T PayloadType;
};

class ACorrelation : public BasicOperation<OutputDataWrapper<>, InputDataWrapper<>, Settings>
{
public:
  static bool IsFloat(const OutputData &out, const InputData &in)
  {
    return (in.GetData1().dataInfo.GetType().Is<std::complex<float>>())
           && (in.GetData2().dataInfo.GetType().Is<std::complex<float>>())
           && (out.GetCorrelations().dataInfo.GetType().Is<std::complex<float>>());
  }

protected:
  virtual bool IsValid(const OutputData &out, const InputData &in, const Settings &) const
  {
    return out.IsValid() && in.IsValid();
  }
};

}// namespace umpalumpa::correlation
