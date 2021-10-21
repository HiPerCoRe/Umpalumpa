#pragma once
/**
 * This is a wrapper for a single Payload
 * It can be used for simple algorithm which do not
 * require multiple Payloads on input or output
 **/
namespace umpalumpa {
namespace data {
  template<typename T> struct SinglePayloadWrapper
  {
    SinglePayloadWrapper(T d) : payload(std::move(d)) {}
    const T payload;
    typedef T PayloadType;

    SinglePayloadWrapper CopyWithoutData() const
    {
      return SinglePayloadWrapper(payload.CopyWithoutData());
    }

    /**
     * Returns true if all Payloads stored here are valid
     **/
    bool IsValid() const { return payload.IsValid(); }

    /**
     * Returns true if all Payloads stored here are equivalent
     * to reference Payloads
     **/
    bool IsEquivalentTo(const SinglePayloadWrapper<T> &ref) const
    {
      return payload.IsEquivalentTo(ref.payload);
    }
  };
}// namespace data
}// namespace umpalumpa