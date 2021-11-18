#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu_kernels.hpp>

namespace umpalumpa::extrema_finder {

namespace {// to avoid poluting
  struct Strategy1 final : public SingleExtremaFinderCPU::Strategy
  {
    // Inherit constructor
    using SingleExtremaFinderCPU::Strategy::Strategy;

    bool Init() override final
    {
      const auto &in = alg.Get().GetInputRef();
      const auto &s = alg.Get().GetSettings();
      return (s.GetVersion() == 1) && (!in.GetData().info.IsPadded())
             && (s.GetLocation() == SearchLocation::kEntire) && (s.GetType() == SearchType::kMax)
             && (s.GetResult() == SearchResult::kValue)
             && (in.GetData().dataInfo.GetType() == umpalumpa::data::DataType::kFloat);
    }

    std::string GetName() const override { return "Strategy1"; }

    bool Execute(const AExtremaFinder::OutputData &out,
      const AExtremaFinder::InputData &in) override
    {
      if (!in.GetData().IsValid() || in.GetData().IsEmpty() || !out.GetValues().IsValid()
          || out.GetValues().IsEmpty())
        return false;
      FindSingleExtremaValXDCPU(reinterpret_cast<float *>(out.GetValues().GetPtr()),
        reinterpret_cast<float *>(in.GetData().GetPtr()),
        in.GetData().info.GetSize(),
        std::greater<float>());
      return true;
    }
  };

  struct Strategy2 final : public SingleExtremaFinderCPU::Strategy
  {
    // Inherit constructor
    using SingleExtremaFinderCPU::Strategy::Strategy;

    bool Init() override
    {
      const auto &in = alg.Get().GetInputRef();
      const auto &s = alg.Get().GetSettings();
      return (s.GetVersion() == 1) && (!in.GetData().info.IsPadded())
             && (s.GetLocation() == SearchLocation::kEntire)
             && (s.GetType() == SearchType::kMax) && (s.GetResult() == SearchResult::kLocation)
             && (in.GetData().dataInfo.GetType() == umpalumpa::data::DataType::kFloat);
    }

    std::string GetName() const override { return "Strategy2"; }

    bool Execute(const AExtremaFinder::OutputData &out,
      const AExtremaFinder::InputData &in) override
    {
      if (!in.GetData().IsValid() || in.GetData().IsEmpty() || !out.GetLocations().IsValid()
          || out.GetLocations().IsEmpty())
        return false;

      // FIXME these values should be read from settings
      // FIXME offset + rectDim cant be > inSize, add check
      // Compute the area to search in
      size_t searchRectWidth = in.GetData().info.GetSize().x;
      size_t searchRectHeight = in.GetData().info.GetSize().y;
      size_t searchRectOffsetX = (in.GetData().info.GetPaddedSize().x - searchRectWidth) / 2;
      size_t searchRectOffsetY = (in.GetData().info.GetPaddedSize().y - searchRectHeight) / 2;

      FindSingleExtremaInRectangle2DCPU<false, true>(
        reinterpret_cast<float *>(out.GetValues().GetPtr()),
        reinterpret_cast<float *>(out.GetLocations().GetPtr()),
        reinterpret_cast<float *>(in.GetData().GetPtr()),
        searchRectOffsetX,
        searchRectOffsetY,
        data::Size(searchRectWidth, searchRectHeight, 1, 1),
        in.GetData().info.GetSize(),
        std::greater<float>());
      return true;
    }
  };
}// namespace

std::vector<std::unique_ptr<SingleExtremaFinderCPU::Strategy>>
  SingleExtremaFinderCPU::GetStrategies() const
{
  std::vector<std::unique_ptr<SingleExtremaFinderCPU::Strategy>> vec;
  vec.emplace_back(std::make_unique<Strategy1>(*this));
  vec.emplace_back(std::make_unique<Strategy2>(*this));
  return vec;
}
}// namespace umpalumpa::extrema_finder
