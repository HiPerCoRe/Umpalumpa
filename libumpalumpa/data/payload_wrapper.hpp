#pragma once

#include <tuple>
#include <cstddef>

namespace umpalumpa::data {
/**
 * This is a wrapper for multiple Payloads
 * Intended usage:
 * Derived class will provide Constructor and Getters
 * for the payloads, so that they can be algorithm-specific
 **/
template<typename... Args> struct PayloadWrapper
{
  /**
   * Returns true if all Payloads stored here are valid
   **/
  bool IsValid() const
  {
    return std::apply([this](const auto &...p) { return ReduceBools(IsValid(p)...); }, payloads);
  }

  /**
   * Returns true if all Payloads stored here are equivalent
   * to reference Payloads
   **/
  bool IsEquivalentTo(const PayloadWrapper<Args...> &ref) const
  {
    return InternalEquivalent(payloads, ref.payloads, std::make_index_sequence<sizeof...(Args)>{});
  }

  /**
   * Intended usage: Derived class can use this method to get std::tuple of Payloads without data
   * to construct new instance.
   */
  auto CopyWithoutData() const
  {
    return std::apply(
      [this](const auto &...p) { return std::make_tuple(RemoveData(p)...); }, payloads);
  }

protected:
  // NOTE not sure if the following constructor is needed
  // All 'args' would have to be copy-constructible
  // PayloadWrapper(const Args &... args) : payload(args...) {}

  PayloadWrapper(Args &&...args) : payloads(std::move(args)...) {}
  PayloadWrapper(std::tuple<Args...> &&t) : payloads(std::move(t)) {}

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
}// namespace umpalumpa::data