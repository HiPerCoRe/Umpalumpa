#pragma once

#include <tuple>
#include <cstddef>
#include <iostream>

namespace umpalumpa::data {
namespace detail {

  /**
   * Black magic that ensures the order of execution when deserializing.
   * The whole Accumulator magic happens during compilation.
   */
  template<typename... Args> struct Accumulator
  {
    constexpr Accumulator() = default;

    template<typename F> constexpr auto operator,(F &&f)
    {
      return std::apply(
        [&f](auto &&... a) { return Accumulator<Args..., decltype(f())>(std::move(a)..., f()); },
        payloads);
    }

    constexpr auto ToTuple() { return std::move(payloads); }

    template<typename T, typename... Rest>
    constexpr Accumulator(T &&t, Rest &&... r) : payloads(std::move(t), std::move(r)...)
    {}

  private:
    std::tuple<Args...> payloads;
  };

  /*
    Might be nice to fix this, but it has some issue with 'using detail::operator,' in the
    PayloadWrapper::Deserialize. ADL finds a wrong overload of operator, while in a
    separate test example it works. :(
    If this worked we wouldn't have to use detail::Accumulator explicitly in the
    PayloadWrapper::Deserialize.
  */
  // template<typename F1, typename F2> constexpr auto operator,(F1 &&f1, F2 &&f2)
  // {
  //   return Accumulator().operator,(std::move(f1)).operator,(std::move(f2));
  // }
}// namespace detail

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
    return std::apply([this](const auto &... p) { return ReduceBools(IsValid(p)...); }, payloads);
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
      [this](const auto &... p) { return std::make_tuple(RemoveData(p)...); }, payloads);
  }

  /**
   * Type of all Payloads
   * Internally we store only references to them
   **/
  typedef std::tuple<Args...> PayloadCollection;

  /**
   * Serializes contained Payloads.
   *
   * Used fold expression guarantees order of execution (extremely important for serialization).
   * TODO ^ should be checked one more time
   */
  void Serialize(std::ostream &out) const
  {
    std::apply([this, &out](const auto &... p) { (..., InternalSerialize(p, out)); }, payloads);
  }

  /**
   * Deserializes Payloads according to the provided template types.
   *
   * The order of execution works with the execution order of the Serialize method.
   */
  static auto Deserialize(std::istream &in)
  {
    return (detail::Accumulator(), ..., [&in]() { return Args::Deserialize(in); }).ToTuple();
  }

protected:
  /**
   * Specific Payload can be accessed using std::get<N>(payloads) function, where N is a position
   * of the requested Payload.
   */
  const std::tuple<Args &...> payloads;// holds references to all Payloads

  /**
   * Create a wrapper of passed Payloads.
   * References to those Payloads are stored, i.e. the wrapper is not taking their ownership
   * Constructor for cases when you want to directly specify Payloads to be wrapped:
   * PayloadWrapper(payload1, payload2);
   **/
  PayloadWrapper(Args &... args) : payloads(args...) {}

  /**
   * Create a wrapper of passed collection of Payloads.
   * References to those Payloads are stored, i.e. the wrapper is not taking their ownership
   * Constructor for cases when you want to pass a std::tuple of Payloads to be wrapped:
   * wrapper 1 = ...
   * auto emptyPayloads = wrapper.CopyWithoutData();
   * auto emptyWrapper = PayloadWrapper(emptyPayloads);
   **/
  PayloadWrapper(std::tuple<Args...> &t) : payloads(MakeTupleRef(t)) {}

private:
  template<std::size_t... Is>
  std::tuple<Args &...> MakeTupleRef(std::tuple<Args...> &tuple, std::index_sequence<Is...>)
  {
    return std::tie(std::get<Is>(tuple)...);
  }

  std::tuple<Args &...> MakeTupleRef(std::tuple<Args...> &tuple)
  {
    return MakeTupleRef(tuple, std::make_index_sequence<sizeof...(Args)>());
  }

  template<typename T> T RemoveData(const T &t) const { return t.CopyWithoutData(); }

  template<typename T> bool IsValid(const T &t) const { return t.IsValid(); }

  template<typename T> bool AreEquivalent(const T &t, const T &ref) const
  {
    return t.IsEquivalentTo(ref);
  }

  template<size_t... I>
  auto InternalEquivalent(const std::tuple<Args &...> &t1,
    const std::tuple<Args &...> &t2,
    std::index_sequence<I...>) const
  {
    return ReduceBools(AreEquivalent(std::get<I>(t1), std::get<I>(t2))...);
  }

  bool ReduceBools(bool b, bool rest...) const { return b && ReduceBools(rest); }

  bool ReduceBools(bool b) const { return b; }

  template<typename T> void InternalSerialize(const T &t, std::ostream &out) const
  {
    t.Serialize(out);
  }
};
}// namespace umpalumpa::data
