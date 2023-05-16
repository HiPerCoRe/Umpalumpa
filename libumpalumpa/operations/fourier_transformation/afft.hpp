#pragma once

#include <complex>
#include <libumpalumpa/operations/basic_operation.hpp>
#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/operations/fourier_transformation/settings.hpp>

namespace umpalumpa::fourier_transformation {

template<typename T> struct OutputDataWrapper : public data::PayloadWrapper<T>
{
  OutputDataWrapper(std::tuple<T> &t) : data::PayloadWrapper<T>(t) {}
  OutputDataWrapper(T &data) : data::PayloadWrapper<T>(data) {}
  T &GetData() const { return std::get<0>(this->payloads); };
  typedef T DataType;
};

template<typename T> struct InputDataWrapper : public data::PayloadWrapper<T>
{
  InputDataWrapper(std::tuple<T> &t) : data::PayloadWrapper<T>(t) {}
  InputDataWrapper(T &data) : data::PayloadWrapper<T>(data) {}
  T &GetData() const { return std::get<0>(this->payloads); };
  typedef T DataType;
};

class AFFT
  : public BasicOperation<OutputDataWrapper<data::Payload<data::FourierDescriptor>>,
      InputDataWrapper<data::Payload<data::FourierDescriptor>>,
      Settings>
{
public:
  static bool IsDouble(const OutputData &out, const InputData &in, Direction d)
  {
    if (Direction::kForward == d) {
      return ((out.GetData().dataInfo.GetType().Is<std::complex<double>>())
              && (in.GetData().dataInfo.GetType().Is<double>()));
    }
    return ((out.GetData().dataInfo.GetType().Is<double>())
            && (in.GetData().dataInfo.GetType().Is<std::complex<double>>()));
  }

  static bool IsFloat(const OutputData &out, const InputData &in, Direction d)
  {
    if (Direction::kForward == d) {
      return ((out.GetData().dataInfo.GetType().Is<std::complex<float>>())
              && (in.GetData().dataInfo.GetType().Is<float>()));
    }
    return ((out.GetData().dataInfo.GetType().Is<float>())
            && (in.GetData().dataInfo.GetType().Is<std::complex<float>>()));
  }

protected:
  bool IsValid(const OutputData &out, const InputData &in, const Settings &s) const override
  {
    return out.GetData().IsValid() && in.GetData().IsValid()
           && (IsDouble(out, in, s.GetDirection()) || IsFloat(out, in, s.GetDirection()));
  }
};
}// namespace umpalumpa::fourier_transformation
