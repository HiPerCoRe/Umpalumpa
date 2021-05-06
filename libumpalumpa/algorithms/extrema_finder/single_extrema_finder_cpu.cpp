#include <functional>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu_kernels.hpp>
#include <libumpalumpa/utils/logger.hpp>

namespace umpalumpa {
namespace extrema_finder {

  namespace {// to avoid poluting
    struct Strategy1 : public SingleExtremaFinderCPU::Strategy
    {
      static constexpr auto kStrategyName = "Strategy1";

      bool Init(const ResultData &, const SearchData &in, const Settings &s) override final
      {
        return (s.version == 1) && (in.info.size == in.info.paddedSize)
               && (s.location == SearchLocation::kEntire) && (s.type == SearchType::kMax)
               && (s.result == SearchResult::kValue)
               && (in.dataInfo.type == umpalumpa::data::DataType::kFloat);
      }

      std::string GetName() const override final { return kStrategyName; }

      bool Execute(const ResultData &out,
        const SearchData &in,
        const Settings &settings) override final
      {
        if (settings.dryRun) return true;
        if (nullptr == in.data || nullptr == out.values->data) return false;
        FindSingleExtremaValXDCPU(reinterpret_cast<float *>(out.values->data),
          reinterpret_cast<float *>(in.data),
          in.info.size,
          std::greater<float>());
        return true;
      }
    };
  }// namespace

  bool SingleExtremaFinderCPU::Init(const ResultData &out,
    const SearchData &in,
    const Settings &settings)
  {
    auto tryToAdd = [this, &out, &in, &settings](auto i) {
      bool canAdd = i->Init(out, in, settings);
      if (canAdd) {
        spdlog::debug("Found valid strategy {}", i->GetName());
        strategy = std::move(i);
      }
      return canAdd;
    };

    return tryToAdd(std::make_unique<Strategy1>()) || false;
  }

  bool SingleExtremaFinderCPU::Execute(const ResultData &out,
    const SearchData &in,
    const Settings &settings)
  {
    if (!this->IsValid(out, in, settings)) return false;
    return strategy->Execute(out, in, settings);
  }
}// namespace extrema_finder
}// namespace umpalumpa
