#pragma once

#include <libumpalumpa/data/multi_payload_wrapper.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include <libumpalumpa/algorithms/correlation/settings.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>

namespace umpalumpa::correlation {

template<typename T = data::Payload<data::FourierDescriptor>>
struct InputDataWrapper : public data::MultiPayloadWrapper<T, T>
{
  InputDataWrapper(std::tuple<T, T> &&t) : data::MultiPayloadWrapper<T,T>(std::move(t)) {}
  InputDataWrapper(T d1, T d2) : data::MultiPayloadWrapper<T, T>(std::move(d1), std::move(d2)) {}
  const T &GetData1() const { return std::get<0>(this->payloads); };
  const T &GetData2() const { return std::get<1>(this->payloads); };
};

template<typename T = data::Payload<data::FourierDescriptor>>
struct OutputDataWrapper : public data::MultiPayloadWrapper<T>
{
  OutputDataWrapper(std::tuple<T> &&t) : data::MultiPayloadWrapper<T>(std::move(t)) {}
  OutputDataWrapper(T correlations) : data::MultiPayloadWrapper<T>(std::move(correlations)) {}
  const T &GetCorrelations() const { return std::get<0>(this->payloads); };
};

class ACorrelation : public BasicAlgorithm<OutputDataWrapper<>, InputDataWrapper<>, Settings>
{
public:
  static bool IsFloat(const OutputData &out, const InputData &in)
  {
    return (in.GetData1().dataInfo.type == data::DataType::kComplexFloat)
           && (in.GetData2().dataInfo.type == data::DataType::kComplexFloat)
           && (out.GetCorrelations().dataInfo.type == data::DataType::kComplexFloat);
  }

protected:
  virtual bool IsValid(const OutputData &out, const InputData &in, const Settings &) const
  {
    return out.IsValid() && in.IsValid();
  }
};

}// namespace umpalumpa::correlation
