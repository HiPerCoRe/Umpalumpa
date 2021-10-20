#pragma once
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>

namespace umpalumpa {
namespace fourier_transformation {
  class FFTCPU : public AFFT
  {
  public:
    bool Init(const OutputData &out, const InputData &in, const Settings &settings) override;
    bool Execute(const OutputData &out, const InputData &in) override;
    void Synchronize() override{};

    struct Strategy
    {
      virtual ~Strategy() = default;
      virtual bool Init(const OutputData &, const InputData &, const Settings &s) = 0;
      virtual bool
        Execute(const OutputData &out, const InputData &in, const Settings &settings) = 0;
      virtual std::string GetName() const = 0;
    };

  private:
    std::unique_ptr<Strategy> strategy;
  };
}// namespace fourier_transformation
}// namespace umpalumpa
