#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <memory>

namespace umpalumpa::extrema_finder {
class SingleExtremaFinderCPU : public AExtremaFinder
{
public:
  bool Init(const OutputData &out, const InputData &in, const Settings &settings) override;
  bool Execute(const OutputData &out, const InputData &in, const Settings &settings) override;
  void Synchronize(){};
  struct Strategy
  {
    virtual ~Strategy() = default;
    virtual bool Init(const OutputData &, const InputData &, const Settings &s) = 0;
    virtual bool Execute(const OutputData &out, const InputData &in, const Settings &settings) = 0;
    virtual std::string GetName() const = 0;
  };

private:
  std::unique_ptr<Strategy> strategy;
};
}// namespace umpalumpa::extrema_finder
