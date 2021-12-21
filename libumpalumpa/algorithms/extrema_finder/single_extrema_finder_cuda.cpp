#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>

namespace umpalumpa::extrema_finder {

namespace {// to avoid poluting
  inline static const auto kKernelFile = utils::GetSourceFilePath(
    "libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda_kernels.cu");

  struct Strategy1 : public SingleExtremaFinderCUDA::KTTStrategy
  {
    // Inherit constructor
    using SingleExtremaFinderCUDA::KTTStrategy::KTTStrategy;

    static constexpr auto kFindMax = "findMax";
    static constexpr auto kRefineLocation = "RefineLocation";

    size_t GetHash() const override { return 0; }
    bool IsEqualTo(const TunableStrategy &ref) const override
    {
      bool isEqual = true;
      try {
        auto &refStrat = dynamic_cast<const Strategy1 &>(ref);
        isEqual = isEqual && GetOutputRef().IsEquivalentTo(refStrat.GetOutputRef());
        isEqual = isEqual && GetInputRef().IsEquivalentTo(refStrat.GetInputRef());
        isEqual = isEqual && GetSettings().IsEquivalentTo(refStrat.GetSettings());
        // Size.n has to be also equal for true equality
        isEqual = isEqual
                  && GetInputRef().GetData().info.GetSize()
                       == refStrat.GetInputRef().GetData().info.GetSize();
      } catch (std::bad_cast &) {
        isEqual = false;
      }
      return isEqual;
    }

    std::unique_ptr<algorithm::Leader> CreateLeader() const override
    {
      return algorithm::StrategyGroup::CreateLeader(*this, alg);
    }

    std::vector<ktt::KernelConfiguration> GetDefaultConfigurations() const override
    {
      return { kttHelper.GetTuner().CreateConfiguration(
                 GetKernelId(), { { "blockSize", static_cast<uint64_t>(32) } }),
        {} };
    }

    bool IsSimilarTo(const TunableStrategy &ref) const override
    {
      bool isSimilar = true;
      try {
        auto &refStrat = dynamic_cast<const Strategy1 &>(ref);
        isSimilar = isSimilar && GetOutputRef().IsEquivalentTo(refStrat.GetOutputRef());
        isSimilar = isSimilar && GetInputRef().IsEquivalentTo(refStrat.GetInputRef());
        isSimilar = isSimilar && GetSettings().IsEquivalentTo(refStrat.GetSettings());
        // Using naive similarity: same as equality except for ignoring Size.n
        // TODO real similarity check
      } catch (std::bad_cast &) {
        isSimilar = false;
      }
      return isSimilar;
    }

    bool InitImpl() override
    {
      const auto &in = alg.Get().GetInputRef();
      const auto &s = alg.Get().GetSettings();
      auto isValidVersion = 1 == s.GetVersion();
      auto isValidLocs = (s.GetResult() == Result::kLocation)
                         && (s.GetLocation() == Location::kEntire)
                         && (s.GetType() == ExtremaType::kMax);
      auto isValidVals = (s.GetResult() == Result::kValue) && (s.GetLocation() == Location::kEntire)
                         && (s.GetType() == ExtremaType::kMax);
      auto isValidData = !in.GetData().info.IsPadded();
      bool canProcess = isValidVersion && isValidData && (isValidLocs || isValidVals);
      if (!canProcess) return false;

      auto &size = in.GetData().info.GetSize();
      auto &tuner = kttHelper.GetTuner();

      // ensure that we have the kernel loaded to KTT
      // this has to be done in critical section, as multiple instances of this algorithm
      // might run on the same worker
      std::lock_guard<std::mutex> lck(kttHelper.GetMutex());
      AddKernelDefinition(kFindMax, kKernelFile, ktt::DimensionVector{ size.n });
      auto definitionId = GetDefinitionId();

      AddKernel(kFindMax, definitionId);
      auto kernelId = GetKernelId();

      if (s.GetPrecision() != Precision::kSingle) {
        auto window = [&s]() {
          switch (s.GetPrecision()) {
          case Precision::k3x3:
            return "3";
          default:
            return "UNSUPPORTED PRECISION";
          }
        }();
        AddKernelDefinition(
          kRefineLocation, kKernelFile, ktt::DimensionVector{ size.n }, { "float", window });
        AddKernel(kRefineLocation, GetDefinitionId(1));
      }

      tuner.AddParameter(kernelId, "blockSize", std::vector<uint64_t>{ 32, 64, 128, 256, 512 });

      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Local,
        ktt::ModifierDimension::X,
        "blockSize",
        ktt::ModifierAction::Multiply);
      return true;
    }

    std::string GetName() const override { return "Strategy1"; }

    bool Execute(const AExtremaFinder::OutputData &out,
      const AExtremaFinder::InputData &in) override
    {
      auto IsFine = [](const auto &p) { return p.IsValid() && !p.IsEmpty(); };
      const auto &s = alg.Get().GetSettings();
      if (!IsFine(in.GetData()) || (!IsFine(out.GetValues()) && !IsFine(out.GetLocations())))
        return false;

      // prepare input data
      auto &tuner = kttHelper.GetTuner();
      auto argIn = AddArgumentVector<float>(in.GetData(), ktt::ArgumentAccessType::ReadOnly);
      auto argSize = tuner.AddArgumentScalar(in.GetData().info.GetSize());

      // prepare output data
      auto argVals = AddArgumentVector<float>(out.GetValues(), ktt::ArgumentAccessType::WriteOnly);
      auto argLocs =
        AddArgumentVector<float>(out.GetLocations(), ktt::ArgumentAccessType::WriteOnly);

      auto definitionId = GetDefinitionId();
      auto kernelId = GetKernelId();

      SetArguments(definitionId, { argIn, argVals, argLocs, argSize });

      auto &size = in.GetData().info.GetSize();
      tuner.SetLauncher(GetKernelId(), [definitionId, &size](ktt::ComputeInterface &interface) {
        auto blockDim = interface.GetCurrentLocalSize(definitionId);
        const ktt::DimensionVector gridDim(size.n);
        interface.RunKernelAsync(definitionId, interface.GetAllQueues().at(0), gridDim, blockDim);
      });

      const bool refine =
        (Result::kLocation == s.GetResult()) && (Precision::k3x3 == s.GetPrecision());
      if (refine) {
        SetArguments(GetDefinitionId(1), { argLocs, argIn, argSize });

        tuner.SetLauncher(GetKernelId(1), [this, &size](ktt::ComputeInterface &interface) {
          const ktt::DimensionVector blockDim(size.n);
          const ktt::DimensionVector gridDim(1);
          interface.RunKernelAsync(
            GetDefinitionId(1), interface.GetAllQueues().at(0), gridDim, blockDim);
        });
      }

      ExecuteKernel(kernelId);
      if (refine) { ExecuteKernel(GetKernelId(1)); }

      return true;
    };
  };

  struct Strategy2 : public SingleExtremaFinderCUDA::KTTStrategy
  {
    // Inherit constructor
    using SingleExtremaFinderCUDA::KTTStrategy::KTTStrategy;

    static constexpr auto kFindMaxRect = "findMaxRect";

    size_t GetHash() const override { return 0; }
    bool IsEqualTo(const TunableStrategy &ref) const override
    {
      bool equal = true;
      try {
        auto &refStrat = dynamic_cast<const Strategy2 &>(ref);
        equal = equal && GetOutputRef().IsEquivalentTo(refStrat.GetOutputRef());
        equal = equal && GetInputRef().IsEquivalentTo(refStrat.GetInputRef());
        equal = equal && GetSettings().IsEquivalentTo(refStrat.GetSettings());
        // Size.n has to be also equal for true equality
        equal = equal
                && GetInputRef().GetData().info.GetSize().n
                     == refStrat.GetInputRef().GetData().info.GetSize().n;
      } catch (std::bad_cast &) {
        equal = false;
      }
      return equal;
    }

    std::unique_ptr<algorithm::Leader> CreateLeader() const override
    {
      return algorithm::StrategyGroup::CreateLeader(*this, alg);
    }

    std::vector<ktt::KernelConfiguration> GetDefaultConfigurations() const override
    {
      return { kttHelper.GetTuner().CreateConfiguration(GetKernelId(),
        { { "blockSizeX", static_cast<uint64_t>(64) },
          { "blockSizeY", static_cast<uint64_t>(2) } }) };
    }

    bool IsSimilarTo(const TunableStrategy &ref) const override
    {
      bool similar = true;
      try {
        auto &refStrat = dynamic_cast<const Strategy2 &>(ref);
        similar = similar && GetOutputRef().IsEquivalentTo(refStrat.GetOutputRef());
        similar = similar && GetInputRef().IsEquivalentTo(refStrat.GetInputRef());
        similar = similar && GetSettings().IsEquivalentTo(refStrat.GetSettings());
        // Using naive similarity: same as equality except for ignoring Size.n
        // TODO real similarity check
      } catch (std::bad_cast &) {
        similar = false;
      }
      return similar;
    }

    bool InitImpl() override
    {
      const auto &in = alg.Get().GetInputRef();
      const auto &s = alg.Get().GetSettings();
      bool canProcess = (s.GetVersion() == 1) && (s.GetLocation() == Location::kRectCenter)
                        && (s.GetType() == ExtremaType::kMax)
                        && (s.GetResult() == Result::kLocation) && (!in.GetData().info.IsPadded())
                        && (in.GetData().dataInfo.GetType() == umpalumpa::data::DataType::kFloat);
      if (!canProcess) return false;

      auto &size = in.GetData().info.GetSize();
      auto &tuner = kttHelper.GetTuner();

      // ensure that we have the kernel loaded to KTT
      // this has to be done in critical section, as multiple instances of this algorithm
      // might run on the same worker
      std::lock_guard<std::mutex> lck(kttHelper.GetMutex());
      AddKernelDefinition(kFindMaxRect, kKernelFile, ktt::DimensionVector{ size.n }, { "float" });
      auto definitionId = GetDefinitionId();

      AddKernel(kFindMaxRect, definitionId);
      auto kernelId = GetKernelId();

      tuner.AddParameter(kernelId, "blockSizeX", std::vector<uint64_t>{ 4, 8, 16, 32, 64, 128 });
      tuner.AddParameter(kernelId, "blockSizeY", std::vector<uint64_t>{ 1, 2, 4, 8, 16, 32, 64 });

      tuner.AddConstraint(
        kernelId, { "blockSizeX", "blockSizeY" }, [&tuner](const std::vector<uint64_t> &params) {
          return params[0] * params[1] <= tuner.GetCurrentDeviceInfo().GetMaxWorkGroupSize();
        });

      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Local,
        ktt::ModifierDimension::X,
        "blockSizeX",
        ktt::ModifierAction::Multiply);
      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Local,
        ktt::ModifierDimension::Y,
        "blockSizeY",
        ktt::ModifierAction::Multiply);

      tuner.SetSearcher(kernelId, std::make_unique<ktt::RandomSearcher>());
      return true;
    }

    std::string GetName() const override { return "Strategy2"; }

    bool Execute(const AExtremaFinder::OutputData &out,
      const AExtremaFinder::InputData &in) override
    {
      if (!in.GetData().IsValid() || in.GetData().IsEmpty() || !out.GetLocations().IsValid()
          || out.GetLocations().IsEmpty())
        return false;

      // prepare input data
      auto &tuner = kttHelper.GetTuner();
      auto argIn = AddArgumentVector<float>(in.GetData(), ktt::ArgumentAccessType::ReadOnly);

      auto argInSize = tuner.AddArgumentScalar(in.GetData().info.GetSize());

      // prepare output data
      auto argVals = tuner.AddArgumentScalar(NULL);
      auto argLocs =
        AddArgumentVector<float>(out.GetLocations(), ktt::ArgumentAccessType::WriteOnly);

      // FIXME these values should be read from settings
      // FIXME offset + rectDim cant be > inSize, add check
      // Compute the area to search in
      size_t searchRectWidth = 28;
      size_t searchRectHeight = 17;
      size_t searchRectOffsetX = (in.GetData().info.GetPaddedSize().x - searchRectWidth) / 2;
      size_t searchRectOffsetY = (in.GetData().info.GetPaddedSize().y - searchRectHeight) / 2;

      auto argOffX = tuner.AddArgumentScalar(searchRectOffsetX);
      auto argOffY = tuner.AddArgumentScalar(searchRectOffsetY);
      auto argRectWidth = tuner.AddArgumentScalar(searchRectWidth);
      auto argRectHeight = tuner.AddArgumentScalar(searchRectHeight);

      auto definitionId = GetDefinitionId();
      auto kernelId = GetKernelId();

      SetArguments(definitionId,
        { argIn, argInSize, argVals, argLocs, argOffX, argOffY, argRectWidth, argRectHeight });

      auto &size = in.GetData().info.GetSize();
      tuner.SetLauncher(kernelId, [definitionId, &size](ktt::ComputeInterface &interface) {
        auto blockDim = interface.GetCurrentLocalSize(definitionId);
        ktt::DimensionVector gridDim(size.n);
        interface.RunKernelAsync(definitionId, interface.GetAllQueues().at(0), gridDim, blockDim);
      });

      ExecuteKernel(kernelId);

      return true;
    };
  };
}// namespace

void SingleExtremaFinderCUDA::Synchronize() { GetHelper().GetTuner().Synchronize(); }

std::vector<std::unique_ptr<SingleExtremaFinderCUDA::Strategy>>
  SingleExtremaFinderCUDA::GetStrategies() const
{
  std::vector<std::unique_ptr<SingleExtremaFinderCUDA::Strategy>> vec;
  vec.emplace_back(std::make_unique<Strategy1>(*this));
  vec.emplace_back(std::make_unique<Strategy2>(*this));
  return vec;
}

}// namespace umpalumpa::extrema_finder
