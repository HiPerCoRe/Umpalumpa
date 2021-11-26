#pragma once

#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/utils/ktt.hpp>

namespace umpalumpa::algorithm {

/**
 * Base class for every strategy that utilizes KTT for tuning.
 *
 * This class is a joining point of the two base classes (BasicAlgorithm::Strategy, TunableStrategy)
 * and it allows their cooperation.
 *
 * Having this class as a predecessor automates many tasks tied to the tuning process.
 */
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

  /**
   * Strategy specific initialization function. Usually used to initialize the KTT tuner.
   */
  virtual bool InitImpl() = 0;

  /**
   * Initialization method automatically called by the BasicAlgorithm class. This overriden version
   * allows to automate some tasks tied to the tuning process.
   */
  bool Init() override final
  {
    TunableStrategy::Cleanup();

    bool initSuccessful = InitImpl();

    if (initSuccessful) { Register(); }
    // TODO maybe some cleanup if not successful? check later

    return initSuccessful;
  }

  template<typename T, typename P>
  auto AddArgumentVector(const data::Payload<P> &p, ktt::ArgumentAccessType at)
  {
    return kttHelper.GetTuner().AddArgumentVector<T>(
      p.GetPtr(), p.info.GetSize().total, at, utils::KTTUtils::GetMemoryNode(p.dataInfo));
  }
};

}// namespace umpalumpa::algorithm
