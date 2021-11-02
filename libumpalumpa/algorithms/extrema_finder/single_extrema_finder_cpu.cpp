#include <functional>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu_kernels.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::extrema_finder {

namespace {// to avoid poluting
  struct Strategy1 : public SingleExtremaFinderCPU::Strategy
  {
    static constexpr auto kStrategyName = "Strategy1";

    bool Init(const AExtremaFinder::OutputData &,
      const AExtremaFinder::InputData &in,
      const Settings &s) override final
    {
      return (s.version == 1) && (!in.data.info.IsPadded())
             && (s.location == SearchLocation::kEntire) && (s.type == SearchType::kMax)
             && (s.result == SearchResult::kValue)
             && (in.data.dataInfo.type == umpalumpa::data::DataType::kFloat);
    }

    std::string GetName() const override final { return kStrategyName; }

    bool Execute(const AExtremaFinder::OutputData &out,
      const AExtremaFinder::InputData &in,
      const Settings &) override final
    {
      if (!in.data.IsValid() || in.data.IsEmpty() || !out.values.IsValid() || out.values.IsEmpty())
        return false;
      FindSingleExtremaValXDCPU(reinterpret_cast<float *>(out.values.ptr),
        reinterpret_cast<float *>(in.data.ptr),
        in.data.info.GetSize(),
        std::greater<float>());
      return true;
    }
  };

  struct Strategy2 : public SingleExtremaFinderCPU::Strategy
  {
    static constexpr auto kStrategyName = "Strategy2";

    bool Init(const AExtremaFinder::OutputData &,
      const AExtremaFinder::InputData &in,
      const Settings &s) override final
    {
      return (s.version == 1) && (!in.data.info.IsPadded())
             && (s.location == SearchLocation::kRectCenter) && (s.type == SearchType::kMax)
             && (s.result == SearchResult::kLocation)
             && (in.data.dataInfo.type == umpalumpa::data::DataType::kFloat);
    }

    std::string GetName() const override final { return kStrategyName; }

    bool Execute(const AExtremaFinder::OutputData &out,
      const AExtremaFinder::InputData &in,
      const Settings &) override final
    {
      if (!in.data.IsValid() || in.data.IsEmpty() || !out.locations.IsValid()
          || out.locations.IsEmpty())
        return false;

      // FIXME these values should be read from settings
      // FIXME offset + rectDim cant be > inSize, add check
      // Compute the area to search in
      size_t searchRectWidth = 28;
      size_t searchRectHeight = 17;
      size_t searchRectOffsetX = (in.data.info.GetPaddedSize().x - searchRectWidth) / 2;
      size_t searchRectOffsetY = (in.data.info.GetPaddedSize().y - searchRectHeight) / 2;

      FindSingleExtremaInRectangle2DCPU<false, true>(reinterpret_cast<float *>(out.values.ptr),
        reinterpret_cast<float *>(out.locations.ptr),
        reinterpret_cast<float *>(in.data.ptr),
        searchRectOffsetX,
        searchRectOffsetY,
        data::Size(searchRectWidth, searchRectHeight, 1, 1),
        in.data.info.GetSize(),
        std::greater<float>());
      return true;
    }
  };
}// namespace

bool SingleExtremaFinderCPU::Init(const OutputData &out,
  const InputData &in,
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

  return tryToAdd(std::make_unique<Strategy1>()) || tryToAdd(std::make_unique<Strategy2>());
}

bool SingleExtremaFinderCPU::Execute(const OutputData &out,
  const InputData &in,
  const Settings &settings)
{
  if (!this->IsValid(out, in, settings)) return false;
  return strategy->Execute(out, in, settings);
}
}// namespace umpalumpa::extrema_finder
