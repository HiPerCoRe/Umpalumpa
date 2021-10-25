#pragma once

#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include <libumpalumpa/data/single_payload_wrapper.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/settings.hpp>
#include <memory>

namespace umpalumpa {
namespace fourier_transformation {
  class AFFT
    : public BasicAlgorithm<data::SinglePayloadWrapper<data::Payload<data::FourierDescriptor>>,
        data::SinglePayloadWrapper<data::Payload<data::FourierDescriptor>>,
        Settings>
  {
  public:
    static bool IsDouble(const OutputData &out, const InputData &in, Direction d)
    {
      if (Direction::kForward == d) {
        return ((out.payload.dataInfo.type == data::DataType::kComplexDouble)
                && (in.payload.dataInfo.type == data::DataType::kDouble));
      }
      return ((out.payload.dataInfo.type == data::DataType::kDouble)
              && (in.payload.dataInfo.type == data::DataType::kComplexDouble));
    }

    static bool IsFloat(const OutputData &out, const InputData &in, Direction d)
    {
      if (Direction::kForward == d) {
        return ((out.payload.dataInfo.type == data::DataType::kComplexFloat)
                && (in.payload.dataInfo.type == data::DataType::kFloat));
      }
      return ((out.payload.dataInfo.type == data::DataType::kFloat)
              && (in.payload.dataInfo.type == data::DataType::kComplexFloat));
    }

  protected:
    bool IsValid(const OutputData &out, const InputData &in, const Settings &s) override
    {
      return out.payload.IsValid() && in.payload.IsValid()
             && (IsDouble(out, in, s.GetDirection()) || IsFloat(out, in, s.GetDirection()));
    }
  };
}// namespace fourier_transformation
}// namespace umpalumpa
