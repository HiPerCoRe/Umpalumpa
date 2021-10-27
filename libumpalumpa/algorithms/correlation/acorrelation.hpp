#pragma once

#include <libumpalumpa/data/multi_payload_wrapper.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/correlation/settings.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>

namespace umpalumpa::correlation {

template<typename T = data::Payload<data::FourierDescriptor>>
struct InputDataWrapper : public data::MultiPayloadWrapper<T, T>
{
  InputDataWrapper(T d1, T d2) : data::MultiPayloadWrapper<T, T>(std::move(d1), std::move(d2)) {}
  const T &GetData1() const { return std::get<0>(this->payloads); };
  const T &GetData2() const { return std::get<1>(this->payloads); };
};

template<typename T = data::Payload<data::FourierDescriptor>>
struct OutputDataWrapper : public data::MultiPayloadWrapper<T>
{
  OutputDataWrapper(T correlations) : data::MultiPayloadWrapper<T>(std::move(correlations)) {}
  const T &GetCorrelations() const { return std::get<0>(this->payloads); };
};

class ACorrelation
{
protected:
  const Settings &GetSettings() const { return *settings.get(); }

  void SetSettings(const Settings &s) { this->settings = std::make_unique<Settings>(s); }

public:
  using OutputData = OutputDataWrapper<>;
  using InputData = InputDataWrapper<>;

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
