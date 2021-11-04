#pragma once

#include <libumpalumpa/data/starpu_utils.hpp>

#include <memory>
namespace umpalumpa {
namespace data {
  template<typename T> class StarpuPayload
  {
  public:
    StarpuPayload &operator=(const StarpuPayload &) = delete;
    StarpuPayload(const StarpuPayload &) =
      delete;// copy prevented to avoid multiple handles to same data; consider using smart pointer

    StarpuPayload(const Payload<T> &p) : handle{ 0 }, payload(p) // FIXME do  we want to take control of the payload completely (e.g. pass it via unique_ptr)
    {
      starpu_payload_register(&handle,
        STARPU_MAIN_RAM,// FIXME this should be taken from the payload
        payload);
      starpu_data_set_name(handle, p.description.c_str());
    }

    ~StarpuPayload()
    {
      if (handle) {
        starpu_data_unregister_submit(handle);// unregister data at some moment
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

    typedef Payload<T> PayloadType;

  private:
    starpu_data_handle_t handle;
    const Payload<T> payload;
  };
}// namespace data
}// namespace umpalumpa