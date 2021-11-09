#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>

namespace umpalumpa::extrema_finder {

namespace {// to avoid poluting
  inline static const auto kKernelFile = utils::GetSourceFilePath(
    "libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda_kernels.cu");

  struct Strategy1 final : public SingleExtremaFinderCUDA::KTTStrategy
  {
    // Inherit constructor
    using SingleExtremaFinderCUDA::KTTStrategy::KTTStrategy;

    static constexpr auto kFindMax1D = "findMax1D";

    size_t GetHash() const override { return 0; }
    bool IsSimilarTo(const TunableStrategy &) const override
    {
      // auto &o = dynamic_cast<const Strategy1 &>(other);
      // TODO real similarity check
      return false;
    }

    bool InitImpl() override
    {
      const auto &in = alg.Get().GetInputRef();
      const auto &s = alg.Get().GetSettings();
      bool canProcess = (s.GetVersion() == 1) && (s.GetLocation() == SearchLocation::kEntire)
                        && (s.GetType() == SearchType::kMax)
                        && (s.GetResult() == SearchResult::kValue)
                        && (!in.GetData().info.IsPadded())
                        && (in.GetData().dataInfo.GetType() == umpalumpa::data::DataType::kFloat);
      if (!canProcess) return false;

      auto &size = in.GetData().info.GetSize();
      auto &tuner = kttHelper.GetTuner();

      // ensure that we have the kernel loaded to KTT
      // this has to be done in critical section, as multiple instances of this algorithm
      // might run on the same worker
      std::lock_guard<std::mutex> lck(kttHelper.GetMutex());
      definitionId = GetKernelDefinitionId(kFindMax1D, kKernelFile, ktt::DimensionVector{ size.n });
      kernelId = tuner.CreateSimpleKernel(kFindMax1D + std::to_string(strategyId), definitionId);

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
      if (!in.GetData().IsValid() || in.GetData().IsEmpty() || !out.GetValues().IsValid()
          || out.GetValues().IsEmpty())
        return false;

      // prepare input data
      auto &tuner = kttHelper.GetTuner();
      auto argIn = tuner.AddArgumentVector<float>(in.GetData().GetPtr(),
        in.GetData().info.GetSize().total,
        ktt::ArgumentAccessType::ReadOnly,
        ktt::ArgumentMemoryLocation::Unified);

      // prepare output data
      auto argVals = tuner.AddArgumentVector<float>(out.GetValues().GetPtr(),
        out.GetValues().info.GetSize().total,
        ktt::ArgumentAccessType::WriteOnly,
        ktt::ArgumentMemoryLocation::Unified);

      auto argSize = tuner.AddArgumentScalar(in.GetData().info.GetSize().single);

      tuner.SetArguments(definitionId, { argIn, argVals, argSize });

      auto &size = in.GetData().info.GetSize();
      tuner.SetLauncher(kernelId, [this, &size](ktt::ComputeInterface &interface) {
        auto blockDim = interface.GetCurrentLocalSize(definitionId);
        const ktt::DimensionVector gridDim(size.n);
        interface.RunKernelAsync(definitionId, interface.GetAllQueues().at(0), gridDim, blockDim);
      });

      if (ShouldTune()) {
        tuner.TuneIteration(kernelId, {});
      } else {
        // TODO GetBestConfiguration can be used once the KTT is able to synchronize
        // the best configuration from multiple KTT instances, or loads the best
        // configuration from previous runs
        // auto bestConfig = tuner.GetBestConfiguration(kernelId);
        auto bestConfig =
          tuner.CreateConfiguration(kernelId, { { "blockSize", static_cast<uint64_t>(32) } });
        tuner.Run(kernelId, bestConfig, {});// run is blocking call
        // arguments shall be removed once the run is done
      }

      return true;
    };
  };

  struct Strategy2 final : public SingleExtremaFinderCUDA::KTTStrategy
  {
    // Inherit constructor
    using SingleExtremaFinderCUDA::KTTStrategy::KTTStrategy;

    static constexpr auto kFindMaxRect = "findMaxRect";

    size_t GetHash() const override { return 0; }
    bool IsSimilarTo(const TunableStrategy &) const override
    {
      // auto &o = dynamic_cast<const Strategy2 &>(other);
      // TODO real similarity check
      return false;
    }

    bool InitImpl() override
    {
      const auto &in = alg.Get().GetInputRef();
      const auto &s = alg.Get().GetSettings();
      bool canProcess = (s.GetVersion() == 1) && (s.GetLocation() == SearchLocation::kRectCenter)
                        && (s.GetType() == SearchType::kMax)
                        && (s.GetResult() == SearchResult::kLocation)
                        && (!in.GetData().info.IsPadded())
                        && (in.GetData().dataInfo.GetType() == umpalumpa::data::DataType::kFloat);
      if (!canProcess) return false;

      auto &size = in.GetData().info.GetSize();
      auto &tuner = kttHelper.GetTuner();

      // ensure that we have the kernel loaded to KTT
      // this has to be done in critical section, as multiple instances of this algorithm
      // might run on the same worker
      std::lock_guard<std::mutex> lck(kttHelper.GetMutex());
      definitionId = GetKernelDefinitionId(
        kFindMaxRect, kKernelFile, ktt::DimensionVector{ size.n }, { "float" });
      kernelId = tuner.CreateSimpleKernel(kFindMaxRect + std::to_string(strategyId), definitionId);

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
      auto argIn = tuner.AddArgumentVector<float>(in.GetData().GetPtr(),
        in.GetData().info.GetSize().total,
        ktt::ArgumentAccessType::ReadOnly,
        ktt::ArgumentMemoryLocation::Unified);

      auto argInSize = tuner.AddArgumentScalar(in.GetData().info.GetSize());

      // prepare output data
      auto argVals = tuner.AddArgumentScalar(NULL);
      auto argLocs = tuner.AddArgumentVector<float>(out.GetLocations().GetPtr(),
        out.GetValues().info.GetSize().total,
        ktt::ArgumentAccessType::WriteOnly,
        ktt::ArgumentMemoryLocation::Unified);

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

      tuner.SetArguments(definitionId,
        { argIn, argInSize, argVals, argLocs, argOffX, argOffY, argRectWidth, argRectHeight });

      auto &size = in.GetData().info.GetSize();
      tuner.SetLauncher(kernelId, [this, &size](ktt::ComputeInterface &interface) {
        auto blockDim = interface.GetCurrentLocalSize(definitionId);
        ktt::DimensionVector gridDim(size.n);
        interface.RunKernelAsync(definitionId, interface.GetAllQueues().at(0), gridDim, blockDim);
      });

      if (ShouldTune()) {
        tuner.TuneIteration(kernelId, {});
      } else {
        // TODO GetBestConfiguration can be used once the KTT is able to synchronize
        // the best configuration from multiple KTT instances, or loads the best
        // configuration from previous runs
        // auto bestConfig = tuner.GetBestConfiguration(kernelId);
        auto bestConfig = tuner.CreateConfiguration(kernelId,
          { { "blockSizeX", static_cast<uint64_t>(64) },
            { "blockSizeY", static_cast<uint64_t>(2) } });
        tuner.Run(kernelId, bestConfig, {});// run is blocking call
        // arguments shall be removed once the run is done
      }

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
