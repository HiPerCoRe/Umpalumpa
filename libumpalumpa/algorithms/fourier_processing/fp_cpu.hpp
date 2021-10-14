#pragma once

#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>
#include <vector>
#include <memory>
#include <map>

namespace umpalumpa {
namespace fourier_processing {
  class FP_CPU : public AFP
  {
  public:
    virtual bool
      Init(const OutputData &out, const InputData &in, const Settings &settings) override;
    virtual bool Execute(const OutputData &out, const InputData &in) override;

    struct Strategy
    {
      virtual ~Strategy() = default;
      virtual bool Init(const OutputData &, const InputData &, const Settings &s) = 0;
      virtual bool
        Execute(const OutputData &out, const InputData &in, const Settings &settings) = 0;
      virtual std::string GetName() const = 0;
    };

    void Synchronize() override {}

  private:
    std::unique_ptr<Strategy> strategy;
  };
}// namespace fourier_processing
}// namespace umpalumpa
