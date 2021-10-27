#pragma once

#include <tuple>

/**
 * This is a wrapper for multiple Payloads
 * It can be used for algorithms which require
 * multiple Payloads on input or output
 **/
namespace umpalumpa {
namespace data {
  template<typename... Args> struct MultiPayloadWrapper
  {
    // NOTE not sure if the following constructor is needed
    // All 'args' would have to be copy-constructible
    // MultiPayloadWrapper(const Args &... args) : payload(args...) {}
    MultiPayloadWrapper(Args &&... args) : payload(std::move(args)...) {}
    const std::tuple<Args...> payload;

    MultiPayloadWrapper CopyWithoutData() const
    {
      return std::apply(
        [this](const auto &... p) { return MultiPayloadWrapper(RemoveData(p)...); }, payload);
    }

    /**
     * Returns true if all Payloads stored here are valid
     **/
    bool IsValid() const
    {
      return std::apply([this](const auto &... p) { return ReduceBools(IsValid(p)...); }, payload);
    }

    /**
     * Returns true if all Payloads stored here are equivalent
     * to reference Payloads
     **/
    bool IsEquivalentTo(const MultiPayloadWrapper<Args...> &ref) const
    {
      return InternalEquivalent(payload, ref.payload, std::make_index_sequence<sizeof...(Args)>{});
    }

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

