#pragma once

#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>

namespace umpalumpa::algorithm {

template<typename O, typename I, typename S>
class KTTStrategyBase
  : public BasicAlgorithm<O, I, S>::Strategy
  , public algorithm::TunableStrategy
{
public:
  // FIXME catch std::bad_cast at a reasonable place and react accordingly
  // Hypothetically, dynamic_cast should always succeed, because all the algorithms that use KTT
  // have to inherit from KTT_Base, and only such algorithms are passed into this constructor
  KTTStrategyBase(const BasicAlgorithm<O, I, S> &algorithm)
    : BasicAlgorithm<O, I, S>::Strategy(algorithm),
      TunableStrategy(dynamic_cast<const KTT_Base &>(algorithm).GetHelper())
  {}

  virtual bool InitImpl() = 0;

  bool Init() override final
  {
    TunableStrategy::Cleanup();

    bool initSuccessful = InitImpl();

    if (initSuccessful) { Register(); }
    // TODO maybe some cleanup if not successful? check later

    return initSuccessful;
  }
};

}// namespace umpalumpa::algorithm

