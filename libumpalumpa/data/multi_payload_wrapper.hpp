#pragma once

#include <tuple>

namespace umpalumpa {
namespace data {
  /**
   * This is a wrapper for multiple Payloads
   * Intended usage:
   * Derived class will provide Constructor and Getters
   * for the payloads, so that they can be algorithm-specific
   **/
  template<typename... Args> struct MultiPayloadWrapper
  {
    /**
     * Returns true if all Payloads stored here are valid
     **/
    bool IsValid() const
    {
      return std::apply([this](const auto &... p) { return ReduceBools(IsValid(p)...); }, payloads);
    }

    /**
     * Returns true if all Payloads stored here are equivalent
     * to reference Payloads
     **/
    bool IsEquivalentTo(const MultiPayloadWrapper<Args...> &ref) const
    {
      return InternalEquivalent(
        payloads, ref.payloads, std::make_index_sequence<sizeof...(Args)>{});
    }

  protected:
    // NOTE not sure if the following constructor is needed
    // All 'args' would have to be copy-constructible
    // MultiPayloadWrapper(const Args &... args) : payload(args...) {}

    MultiPayloadWrapper(Args &&... args) : payloads(std::move(args)...) {}

    /**
     * Intended usage: Derived class can use this method to get its copy without data.
     *
     * Example for derived class Derived:
     *
     * MultiPayloadWrapper<Args...>::template CopyWithoutData<Derived>();
     */
    template<typename T> T CopyWithoutData() const
    {
      return std::apply([this](const auto &... p) { return T(RemoveData(p)...); }, payloads);
    }

    /**
     * Specific Payload can be accessed using std::get<N>(payloads) function, where N is a position
     * of the requested Payload.
     */
    const std::tuple<Args...> payloads;

  private:
    template<typename T> T RemoveData(const T &t) const { return t.CopyWithoutData(); }
    template<typename T> bool IsValid(const T &t) const { return t.IsValid(); }

    template<typename T> bool AreEquivalent(const T &t, const T &ref) const
    {
      return t.IsEquivalentTo(ref);
    }

    template<size_t... I>
    auto InternalEquivalent(const std::tuple<Args...> &t1,
      const std::tuple<Args...> &t2,
      std::index_sequence<I...>) const
    {
      return ReduceBools(AreEquivalent(std::get<I>(t1), std::get<I>(t2))...);
    }

    bool ReduceBools(bool b, bool rest...) const { return b && ReduceBools(rest); }
    bool ReduceBools(bool b) const { return b; }
  };
}// namespace data
}// namespace umpalumpa

