#pragma once

#include <libumpalumpa/data/starpu_utils.hpp>

#include <iostream>

#include <memory>
namespace umpalumpa {
namespace data {
  template<typename T> class StarpuPayload
  {
  public:
    StarpuPayload(const StarpuPayload &) =
      delete;// copy prevented to avoid multiple handles to same data; consider using smart pointer

    StarpuPayload(const Payload<T> &p) : handle{ 0 }, payload(p)
    {
      starpu_payload_register(&handle,
        STARPU_MAIN_RAM,// FIXME this should be taken from the payload
        p);
      starpu_data_set_name(handle, p.description.c_str());
    }

    StarpuPayload() : handle{ 0 }, payload{} { starpu_void_data_register(&handle); }

    ~StarpuPayload()
    {
      if (handle) {
        starpu_data_unregister_submit(handle);// unregister data at leasure
      }
    }

    void Unregister(bool copyToHomeNode = true)
    {
      if (copyToHomeNode) {
        starpu_data_unregister(handle);// blocking call
      } else {
        starpu_data_unregister_no_coherency(handle);
      }
      handle = { 0 };
    }

    const auto &GetHandle() const { return handle; }
    const auto &GetPayload() const { return payload; }

  private:
    starpu_data_handle_t handle;
    const Payload<T> payload;
  };
}// namespace data
}// namespace umpalumpa