#pragma once

#include <memory>
#include <vector>

namespace umpalumpa {
template<typename O, typename I, typename S> class BasicAlgorithm
{
public:
  typedef O OutputData;
  typedef I InputData;
  typedef S Settings;

  virtual ~BasicAlgorithm() { this->Cleanup(); }

  /**
   * Initialize this algorithm.
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
    // test that input is not complete garbage
    if (!this->IsValid(out, in, s)) { return false; };
    // store information about init
    this->outputRef = std::make_unique<OutputData>(out.CopyWithoutData());
    this->inputRef = std::make_unique<InputData>(in.CopyWithoutData());
    this->settings = std::make_unique<Settings>(s);
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
   * Execute this algorithm.
   * Execution can happen only if:
   *  - the algorithm has been already successfuly initialized
   *  - the output and input data are similar, except for the N
   *  - the output and input data are valid
   *  - the output and input data do not break additional requirements
   * Derived classes can override ExecuteImpl if they require different behavior.
   **/
  [[nodiscard]] bool Execute(const OutputData &out, const InputData &in)
  {
    bool canExecute = this->IsInitialized() && out.IsEquivalentTo(this->GetOutputRef())
                      && in.IsEquivalentTo(this->GetInputRef()) && out.IsValid() && in.IsValid();
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

protected:
  struct Strategy
  {
    virtual ~Strategy() = default;
    virtual bool Init(const OutputData &, const InputData &, const Settings &s) = 0;
    virtual bool Execute(const OutputData &out, const InputData &in, const Settings &settings) = 0;
    virtual std::string GetName() const = 0;
  };

  /**
   * Returns true if output, input and settings are not logically conflicting or malformed.
   **/
  virtual bool IsValid(const OutputData &out, const InputData &in, const Settings &s) = 0;

  /**
   * Perform specific initialization of the algorithm.
   * Derived classes are encouraged to override this method.
   * This basic version assumes that derived class provides possible Strategies,
   * so it's trying to initialize the first one of them.
   **/
  virtual bool InitImpl()
  {
    const auto &out = this->GetOutputRef();
    const auto &in = this->GetInputRef();
    const auto &s = this->GetSettings();
    auto tryToInit = [this, &out, &in, &s](auto &i) {
      bool canUse = i->Init(out, in, s);
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
   * Perform specific execution of the algorithm.
   * Derived classes are encouraged to override this method.
   * This basic version assumes that working Strategy has been stored
   * by the default InitImpl() method.
   * No addional checks are performed.
   **/
  virtual bool ExecuteImpl(const OutputData &out, const InputData &in)
  {
    return strategy->Execute(out, in, this->GetSettings());
  }

  /**
   * This method should provide available Strategies for executing this
   * algorithm. By default, the first Strategy which returns true from its
   * initialization method will be used during the execution.
   **/
  virtual std::vector<std::unique_ptr<Strategy>> GetStrategies() const { return {}; };

  const OutputData &GetOutputRef() const { return *outputRef.get(); }

  const InputData &GetInputRef() const { return *inputRef.get(); }

  const Settings &GetSettings() const { return *settings.get(); }


private:
  std::unique_ptr<OutputData> outputRef;
  std::unique_ptr<InputData> inputRef;
  std::unique_ptr<Settings> settings;
  std::unique_ptr<Strategy> strategy;
  bool isInitialized = false;
};
}// namespace umpalumpa