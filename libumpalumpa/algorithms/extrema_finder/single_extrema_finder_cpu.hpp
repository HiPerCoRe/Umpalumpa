#pragma once

#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>
#include <memory>

namespace umpalumpa {
namespace extrema_finder {
  class SingleExtremaFinderCPU : public AExtremaFinder
  {
  public:
    bool Init(const ResultData &out, const SearchData &in, const Settings &settings) override;
    bool Execute(const ResultData &out, const SearchData &in, const Settings &settings) override;
    void Synchronize(){};
    struct Strategy
    {
      virtual ~Strategy() = default;
      virtual bool Init(const ResultData &, const SearchData &, const Settings &s) = 0;
      virtual bool
        Execute(const ResultData &out, const SearchData &in, const Settings &settings) = 0;
      virtual std::string GetName() const = 0;
    };

  private:
    std::unique_ptr<Strategy> strategy;
  };
}// namespace extrema_finder
}// namespace umpalumpa
