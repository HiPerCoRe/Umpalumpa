#pragma once

#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/fourier_processing/settings.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>

namespace umpalumpa {
namespace fourier_processing {
  class AFP 
  {
  protected:
    template<typename T, typename U> struct InputDataWrapper
    {
      InputDataWrapper(T &&d, U &&f) : data(std::move(d)), filter(std::move(f)) {}
      const T data;
      const U filter;
    };

    template<typename T> struct OutputDataWrapper
    {
      OutputDataWrapper(T &&d) : data(std::move(d)) {}
      const T data;
    };
 
    const Settings& GetSettings() const {
      return *settings.get();
    }

    void SetSettings(const Settings& s) {
      this->settings = std::make_unique<Settings>(s);
    }

  public:
    using OutputData = OutputDataWrapper<data::Payload<data::FourierDescriptor>>;
    using InputData = InputDataWrapper<data::Payload<data::FourierDescriptor>, data::Payload<data::LogicalDescriptor>>;

    virtual bool Init(const OutputData &out, const InputData &in, const Settings &settings) = 0;
    virtual bool Execute(const OutputData &out, const InputData &in) = 0;
    virtual void Cleanup(){};
    virtual void Synchronize() = 0;

    virtual ~AFP() = default;

  protected:
    virtual bool
      IsValid(const OutputData &, const InputData &) const // move to cpp
    {
      bool result = true;
      return result;
    }

    std::unique_ptr<Settings> settings;
  };

}// namespace fourier_processing
}// namespace umpalumpa
