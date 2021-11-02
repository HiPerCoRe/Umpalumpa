#pragma once

#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/settings.hpp>
#include <memory>

namespace umpalumpa::fourier_transformation {

template<typename T> struct OutputDataWrapper : public data::PayloadWrapper<T>
{
  OutputDataWrapper(std::tuple<T> &&t) : data::PayloadWrapper<T>(std::move(t)) {}
  OutputDataWrapper(T data) : data::PayloadWrapper<T>(std::move(data)) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
  typedef T PayloadType;
};

template<typename T> struct InputDataWrapper : public data::PayloadWrapper<T>
{
  InputDataWrapper(std::tuple<T> &&t) : data::PayloadWrapper<T>(std::move(t)) {}
  InputDataWrapper(T data) : data::PayloadWrapper<T>(std::move(data)) {}
  const T &GetData() const { return std::get<0>(this->payloads); };
  typedef T PayloadType;
};

class AFFT
  : public BasicAlgorithm<OutputDataWrapper<data::Payload<data::FourierDescriptor>>,
      InputDataWrapper<data::Payload<data::FourierDescriptor>>,
      Settings>
{
public:
  static bool IsDouble(const OutputData &out, const InputData &in, Direction d)
  {
    if (Direction::kForward == d) {
      return ((out.GetData().dataInfo.type == data::DataType::kComplexDouble)
              && (in.GetData().dataInfo.type == data::DataType::kDouble));
    }
    return ((out.GetData().dataInfo.type == data::DataType::kDouble)
            && (in.GetData().dataInfo.type == data::DataType::kComplexDouble));
  }

  static bool IsFloat(const OutputData &out, const InputData &in, Direction d)
  {
    if (Direction::kForward == d) {
      return ((out.GetData().dataInfo.type == data::DataType::kComplexFloat)
              && (in.GetData().dataInfo.type == data::DataType::kFloat));
    }
    return ((out.GetData().dataInfo.type == data::DataType::kFloat)
            && (in.GetData().dataInfo.type == data::DataType::kComplexFloat));
  }

protected:
  bool IsValid(const OutputData &out, const InputData &in, const Settings &s) const override
  {
    return out.GetData().IsValid() && in.GetData().IsValid()
           && (IsDouble(out, in, s.GetDirection()) || IsFloat(out, in, s.GetDirection()));
  }
};
}// namespace umpalumpa::fourier_transformation
