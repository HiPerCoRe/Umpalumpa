#include <libumpalumpa/operations/extrema_finder/single_extrema_finder_cpu.hpp>
#include <libumpalumpa/operations/extrema_finder/single_extrema_finder_cpu_kernels.hpp>

namespace umpalumpa::extrema_finder {

namespace {// to avoid poluting
  struct Strategy1 final : public SingleExtremaFinderCPU::Strategy
  {
    // Inherit constructor
    using SingleExtremaFinderCPU::Strategy::Strategy;

    bool Init() override final
    {
      const auto &in = op.Get().GetInputRef();
      const auto &s = op.Get().GetSettings();
      auto isValidVersion = 1 == s.GetVersion();
      auto isValidLocs = (s.GetResult() == Result::kLocation)
                         && (s.GetLocation() == Location::kEntire)
                         && (s.GetType() == ExtremaType::kMax);
      auto isValidVals = (s.GetResult() == Result::kValue) && (s.GetLocation() == Location::kEntire)
                         && (s.GetType() == ExtremaType::kMax);
      auto isValidData = !in.GetData().info.IsPadded();
      return isValidVersion && isValidData && (isValidLocs || isValidVals);
    }

    std::string GetName() const override { return "Strategy1"; }

    bool Execute(const AExtremaFinder::OutputData &out,
      const AExtremaFinder::InputData &in) override
    {
      auto IsFine = [](const auto &p) { return p.IsValid() && !p.IsEmpty(); };
      const auto &s = op.Get().GetSettings();
      if (!IsFine(in.GetData()) || (!IsFine(out.GetValues()) && !IsFine(out.GetLocations())))
        return false;
      switch (s.GetResult()) {
      case Result::kValue:
        FindSingleExtremaCPU<true, false>(reinterpret_cast<float *>(out.GetValues().GetPtr()),
          nullptr,
          reinterpret_cast<float *>(in.GetData().GetPtr()),
          in.GetData().info.GetSize(),
          std::greater<float>());
        break;
      case Result::kLocation:
        FindSingleExtremaCPU<false, true, float>(nullptr,
          reinterpret_cast<float *>(out.GetLocations().GetPtr()),
          reinterpret_cast<float *>(in.GetData().GetPtr()),
          in.GetData().info.GetSize(),
          std::greater<float>());
        break;
      case Result::kBoth:
        FindSingleExtremaCPU<true, true>(reinterpret_cast<float *>(out.GetValues().GetPtr()),
          reinterpret_cast<float *>(out.GetLocations().GetPtr()),
          reinterpret_cast<float *>(in.GetData().GetPtr()),
          in.GetData().info.GetSize(),
          std::greater<float>());
        break;
      default:
        return false;
      }
      if ((Result::kLocation == s.GetResult()) && (Precision::k3x3 == s.GetPrecision())) {
        RefineLocation<float, 3>(reinterpret_cast<float *>(out.GetLocations().GetPtr()),
          reinterpret_cast<float *>(in.GetData().GetPtr()),
          in.GetData().info.GetSize());
      }
      return true;
    }
  };

  struct Strategy2 final : public SingleExtremaFinderCPU::Strategy
  {
    // Inherit constructor
    using SingleExtremaFinderCPU::Strategy::Strategy;

    bool Init() override
    {
      const auto &in = op.Get().GetInputRef();
      const auto &s = op.Get().GetSettings();
      return (s.GetVersion() == 1) && (!in.GetData().info.IsPadded())
             && (s.GetLocation() == Location::kRectCenter) && (s.GetType() == ExtremaType::kMax)
             && (s.GetResult() == Result::kLocation)
             && (in.GetData().dataInfo.GetType().Is<float>());
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
      size_t searchRectWidth = 28;
      size_t searchRectHeight = 17;
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
