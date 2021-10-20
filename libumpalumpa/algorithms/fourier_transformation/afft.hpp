#pragma once

#include "libumpalumpa/data/fourier_descriptor.hpp"
#include "libumpalumpa/data/payload.hpp"
#include <libumpalumpa/algorithms/fourier_transformation/settings.hpp>
#include <memory>

namespace umpalumpa {
namespace fourier_transformation {
  class AFFT 
  {
  protected:
    template<typename T> struct DataWrapper
    {
      DataWrapper(T &&d) : data(std::move(d)) {}
      const T data;
      typedef T PayloadType;
    };

    const Settings& GetSettings() const {
      return *settings.get();
    }

    void SetSettings(const Settings& s) {
      this->settings = std::make_unique<Settings>(s);
    }

    // FIXME IsValid needs to check the data type (either float or double)

  public:
    using OutputData = DataWrapper<data::Payload<data::FourierDescriptor>>;
    using InputData = DataWrapper<data::Payload<data::FourierDescriptor>>;
    virtual bool Init(const OutputData &out, const InputData &in, const Settings &settings) = 0;
    virtual bool Execute(const OutputData &out, const InputData &in) = 0;
    virtual void Cleanup(){
      settings.reset();
    };
    virtual void Synchronize() = 0;

    bool IsInitialized() { return settings.get(); }

    virtual ~AFFT() = default;

  private:
    std::unique_ptr<Settings> settings;
  };
}
}
