#pragma once

#include <memory>
#include <vector>

namespace umpalumpa {
template<typename O, typename I, typename S> class BasicOperation
{
public:
  typedef O OutputData;
  typedef I InputData;
  typedef S Settings;

  virtual ~BasicOperation() { this->Cleanup(); }

  /**
   * Initialize this operation.
   * Initialization removes any previous state, i.e. calling Init() is
   * equivalent to calling Cleanup() followed by Init();
   * In the first step, some basic checks are done. If passed, then
   * copies of the Output and Input Payloads (without data), as well as Settings are
   * locally stored for future reference.
   * These copies can be accessed by derived classes.
   * Derived classes can override InitImpl if they require different behavior.
   * Returns true if initialization happened correctly.
   * If false is returned, the internal state of the instance is undefined.
   **/
  [[nodiscard]] bool Init(const OutputData &out, const InputData &in, const Settings &s)
  {
    // get rid of any previous state
    this->Cleanup();
    // create reference payloads without data
    this->emptyOutputPayloads = std::make_unique<OPC>(out.CopyWithoutData());
    this->outputRef = std::make_unique<OutputData>(*emptyOutputPayloads);
    this->emptyInputPayloads = std::make_unique<IPC>(in.CopyWithoutData());
    this->inputRef = std::make_unique<InputData>(*emptyInputPayloads);
    this->settings = std::make_unique<Settings>(s);
    // test that input is not complete garbage
    if (!this->IsValid(*outputRef, *inputRef, s)) { return false; };
    // check if we have working implementation
    if (this->InitImpl()) {
      this->isInitialized = true;
      return true;
    };
    // init failed
    this->Cleanup();
    return false;
  }

  /**
   * Execute this Operation.
   * Execution can happen only if:
   *  - the Operation has been already successfuly initialized
   *  - the output and input data are similar, except for the N
   *  - the output and input data are valid
   *  - the output and input data do not break additional requirements
   * Notice that Operation itself decides if the input / output data are valid
   * Derived classes can override ExecuteImpl if they require different behavior.
   **/
  [[nodiscard]] bool Execute(const OutputData &out, const InputData &in)
  {
    bool canExecute = this->IsInitialized() && out.IsEquivalentTo(this->GetOutputRef())
                      && in.IsEquivalentTo(this->GetInputRef());
    if (canExecute) { return this->ExecuteImpl(out, in); }
    return false;
  }

  /**
   * Clean all data hold in this instance.
   * If overloaded, derived class should call this implementation too.
   **/
  virtual void Cleanup()
  {
    this->outputRef.reset();
    this->inputRef.reset();
    this->settings.reset();
    this->strategy.reset();
    this->isInitialized = false;
  }

  virtual void Synchronize() = 0;

  bool IsInitialized() const { return isInitialized; }

  /**
   * Method to obtain reference to this instance.
   * It is meant to be used by Strategies to get additional
   * information stored in the Operation.
   * By using Covariant return types, derived classes can
   * override this method and thus provide more specific
   * type for their Strategies.
   **/
  virtual const BasicOperation &Get() const { return *this; }

  const OutputData &GetOutputRef() const { return *outputRef.get(); }

  const InputData &GetInputRef() const { return *inputRef.get(); }

  const Settings &GetSettings() const { return *settings.get(); }

  /**
   * Return bytes internally allocated by the operation to be able to process the
   * given work
   **/
  virtual size_t GetUsedBytes() const { return 0; };

  struct Strategy
  {
    Strategy(const BasicOperation &a) : op(a) {}
    virtual ~Strategy() = default;
    virtual bool Init() = 0;
    virtual bool Execute(const OutputData &out, const InputData &in) = 0;
    virtual std::string GetName() const = 0;

  protected:
    const BasicOperation &op;
  };

  /**
   * FIXME Just a temporary solution to get access to a strategy.
   * will be substituted by some TuningHint/TuningSettings struct
   **/
  Strategy &GetStrategy() { return *strategy; }

protected:
  /**
   * Returns true if output, input and settings are not logically conflicting or malformed.
   **/
  virtual bool IsValid(const OutputData &out, const InputData &in, const Settings &s) const = 0;

  /**
   * Perform specific initialization of the operation.
   * Derived classes are encouraged to override this method.
   * This basic version assumes that derived class provides possible Strategies,
   * so it's trying to initialize the first one of them.
   **/
  virtual bool InitImpl()
  {
    auto tryToInit = [this](auto &i) {
      bool canUse = i->Init();
      if (canUse) { strategy = std::move(i); }
      return canUse;
    };
    auto availableStrategies = this->GetStrategies();
    for (auto &str : availableStrategies) {
      if (tryToInit(str)) return true;
    }
    return false;
  };

  /**
   * Perform specific execution of the operation.
   * Derived classes are encouraged to override this method.
   * This basic version assumes that working Strategy has been stored
   * by the default InitImpl() method.
   * No addional checks are performed.
   **/
  virtual bool ExecuteImpl(const OutputData &out, const InputData &in)
  {
    return strategy->Execute(out, in);
  }

  /**
   * This method should provide available Strategies for executing this
   * operation. By default, the first Strategy which returns true from its
   * initialization method will be used during the execution.
   **/
  virtual std::vector<std::unique_ptr<Strategy>> GetStrategies() const { return {}; };

private:
  typedef typename O::PayloadCollection OPC;
  typedef typename I::PayloadCollection IPC;

  std::unique_ptr<OPC> emptyOutputPayloads;
  std::unique_ptr<IPC> emptyInputPayloads;
  std::unique_ptr<OutputData> outputRef;
  std::unique_ptr<InputData> inputRef;
  std::unique_ptr<Settings> settings;
  std::unique_ptr<Strategy> strategy;
  bool isInitialized = false;
};
}// namespace umpalumpa
