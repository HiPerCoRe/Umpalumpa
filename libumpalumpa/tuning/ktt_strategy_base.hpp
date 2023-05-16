#pragma once

#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/operations/basic_operation.hpp>
#include <libumpalumpa/tuning/ktt_base.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/utils/ktt.hpp>
#include <sstream>

namespace umpalumpa::tuning {

/**
 * Base class for every strategy that utilizes KTT for tuning.
 *
 * This class is a joining point of the two base classes (BasicOperation::Strategy, TunableStrategy)
 * and it allows their cooperation.
 *
 * Having this class as a predecessor automates many tasks tied to the tuning process.
 */
template<typename O, typename I, typename S>
class KTTStrategyBase
  : public BasicOperation<O, I, S>::Strategy
  , public tuning::TunableStrategy
{
public:
  using StrategyOutput = O;
  using StrategyInput = I;
  using StrategySettings = S;

  // FIXME catch std::bad_cast at a reasonable place and react accordingly
  // Hypothetically, dynamic_cast should always succeed, because all the operations that use KTT
  // have to inherit from KTT_Base, and only such operations are passed into this constructor
  KTTStrategyBase(const BasicOperation<O, I, S> &operation)
    : BasicOperation<O, I, S>::Strategy(operation),
      TunableStrategy(dynamic_cast<const KTT_Base &>(operation).GetHelper())
  {}

  /**
   * Initialization method automatically called by the BasicOperation class. This overriden version
   * allows to automate some tasks tied to the tuning process.
   */
  bool Init() override final
  {
    TunableStrategy::Cleanup();

    bool initSuccessful = InitImpl();

    if (initSuccessful) {
      SetUniqueStrategyName();// This has to be called before the Register()!!!
      Register();
    }
    // TODO maybe some cleanup if not successful? check later

    return initSuccessful;
  }

  /**
   * Strategy specific initialization function. Usually used to initialize the KTT tuner.
   */
  virtual bool InitImpl() = 0;

  /**
   * Getter for operation's OutputData.
   */
  virtual const StrategyOutput &GetOutputRef() const
  {
    return BasicOperation<O, I, S>::Strategy::op.GetOutputRef();
  }

  /**
   * Getter for operation's InputData.
   */
  virtual const StrategyInput &GetInputRef() const
  {
    return BasicOperation<O, I, S>::Strategy::op.GetInputRef();
  }

  /**
   * Getter for operation's Settings.
   */
  virtual const StrategySettings &GetSettings() const
  {
    return BasicOperation<O, I, S>::Strategy::op.GetSettings();
  }

  std::string GetUniqueName() const final
  {
    if (uniqueStrategyName.empty()) {
      throw std::logic_error("Access to uninitialized 'unique strategy name'!");
    }
    return uniqueStrategyName;
  }

protected:
  /**
   * Creates a KTT argument of type Vector from the content of the Payload.
   * If the Payload is empty, NULL argument will be returned.
   **/
  template<typename T, typename P>
  auto AddArgumentVector(const data::Payload<P> &p, ktt::ArgumentAccessType at)
  {
    if (p.IsEmpty()) { return kttHelper.GetTuner().AddArgumentScalar(NULL); }
    return kttHelper.GetTuner().template AddArgumentVector<T>(
      p.GetPtr(), p.info.GetSize().total, at, utils::KTTUtils::GetMemoryNode(p.dataInfo));
  }

  /**
   * Creates and sets unique strategy name based on the OutputData, InputData, Settings.
   */
  void SetUniqueStrategyName()
  {
    std::stringstream ss;
    std::stringstream unique;
    unique << kttHelper.GetTuner().GetCurrentDeviceInfo().GetName() << '-';
    unique << GetFullName();
    unique << '-';
    GetOutputRef().Serialize(ss);
    unique << std::hash<std::string>{}(ss.str());
    ss.clear();
    unique << '-';
    GetInputRef().Serialize(ss);
    unique << std::hash<std::string>{}(ss.str());
    ss.clear();
    unique << '-';
    GetSettings().Serialize(ss);
    unique << std::hash<std::string>{}(ss.str());
    // std::stringstream noWhitespaces;
    // while (!ss.eof()) {
    //   std::string tmp;
    //   ss >> tmp;
    //   noWhitespaces << tmp;
    // }
    uniqueStrategyName = unique.str();
  }

private:
  std::string uniqueStrategyName;
};

}// namespace umpalumpa::tuning
