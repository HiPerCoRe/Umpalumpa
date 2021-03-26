#include <functional>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu_kernels.hpp>

namespace umpalumpa {
namespace extrema_finder {

  namespace {// to avoid poluting
    struct Strategy1
    {
      static bool CanRun(__attribute__((unused)) const ResultData &out, const SearchData &in, const Settings &settings)
      {
        return (settings.version == 1) && (in.info.size == in.info.paddedSize)
               && (settings.location == SearchLocation::kEntire) && (settings.type == SearchType::kMax)
               && (settings.result == SearchResult::kValue) && (in.dataInfo.type == umpalumpa::data::DataType::kFloat);
      }

      static bool Run(const ResultData &out, const SearchData &in, const Settings &settings)
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

  bool SingleExtremaFinderCPU::Execute(const ResultData &out, const SearchData &in, const Settings &settings)
  {
    if (!this->IsValid(out, in, settings)) return false;
    if (Strategy1::CanRun(out, in, settings)) return Strategy1::Run(out, in, settings);
    return false;// no strategy could process these data
  }
}// namespace extrema_finder
}// namespace umpalumpa
