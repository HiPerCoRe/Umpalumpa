#pragma once

#include "libumpalumpa/data/fourier_descriptor.hpp"
#include "libumpalumpa/data/payload.hpp"
#include <libumpalumpa/algorithms/fourier_transformation/settings.hpp>

namespace umpalumpa {
namespace fourier_transformation {
  class AFFT 
  {
  protected:
    template<typename T> struct DataWrapper
    {
      DataWrapper(T &&d) : data(std::move(d)) {}
      const T data;
      typedef T type;
    };

    const Settings& GetSettings() const {
      return settings;
    }

    void SetSettings(const Settings& settings) {
      this->settings = settings;
    }

    // FIXME IsValid needs to check the data type (either float or double)

  public:
    using ResultData = DataWrapper<data::Payload<data::FourierDescriptor>>;
    using InputData = DataWrapper<data::Payload<data::FourierDescriptor>>;
    virtual bool Init(const ResultData &out, const InputData &in, const Settings &settings) = 0;
    virtual bool Execute(const ResultData &out, const InputData &in) = 0;
    virtual void Cleanup(){};
    virtual void Synchronize() = 0;


    virtual ~AFFT() = default;

  private:
    Settings settings;
  };
}
}
