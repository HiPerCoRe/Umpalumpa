#pragma once

#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/correlation/settings.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>

namespace umpalumpa::correlation {
class ACorrelation
{
protected:
  template<typename T> struct InputDataWrapper
  {
    InputDataWrapper(T &&d1, T &&d2) : data1(std::move(d1)), data2(std::move(d2)) {}
    const T data1;
    const T data2;
  };

  template<typename T> struct OutputDataWrapper
  {
    OutputDataWrapper(T &&d) : data(std::move(d)) {}
    const T data;
  };

  const Settings &GetSettings() const { return *settings.get(); }

  void SetSettings(const Settings &s) { this->settings = std::make_unique<Settings>(s); }

public:
  using OutputData = OutputDataWrapper<data::Payload<data::FourierDescriptor>>;
  using InputData = InputDataWrapper<data::Payload<data::FourierDescriptor>>;

  virtual bool Init(const OutputData &out, const InputData &in, const Settings &settings) = 0;
  virtual bool Execute(const OutputData &out, const InputData &in) = 0;
  virtual void Cleanup(){};
  virtual void Synchronize() = 0;

  virtual ~ACorrelation() = default;

protected:
  virtual bool IsValid(const OutputData &, const InputData &) const// move to cpp
  {
    bool result = true;
    return result;
  }

  std::unique_ptr<Settings> settings;
};

}// namespace umpalumpa::correlation
