#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/utils/system.hpp>
#include <libumpalumpa/utils/cuda.hpp>

namespace umpalumpa {
namespace extrema_finder {

  namespace {// to avoid poluting
    inline static const auto kKernelFile = utils::GetSourceFilePath(
      "libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda_kernels.cu");

    struct Strategy1 : public SingleExtremaFinderCUDA::Strategy
    {
      static constexpr auto kFindMax1D = "findMax1D";

      size_t GetHash() const override { return 0; }
      bool IsSimilarTo(const TunableStrategy &ref) const override
      {
        if (GetFullName() != ref.GetFullName()) { return false; }
        // Now we know that type of 'other' is the same as 'this' and we can safely cast it to the
        // needed type
        // auto &o = dynamic_cast<const Strategy1 &>(other);
        // TODO real similarity check
        return false;
      }

      bool Init(const AExtremaFinder::OutputData &,
        const AExtremaFinder::InputData &in,
        const Settings &s,
        utils::KTTHelper &helper) override final
      {
        bool canProcess = (s.version == 1) && (s.location == SearchLocation::kEntire)
                          && (s.type == SearchType::kMax) && (s.result == SearchResult::kValue)
                          && (!in.data.info.IsPadded())
                          && (in.data.dataInfo.type == umpalumpa::data::DataType::kFloat);
        if (canProcess) {
          TunableStrategy::Init(helper);
          auto &size = in.data.info.GetSize();
          auto &tuner = helper.GetTuner();

          // ensure that we have the kernel loaded to KTT
          // this has to be done in critical section, as multiple instances of this algorithm
          // might run on the same worker
          std::lock_guard<std::mutex> lck(helper.GetMutex());
          definitionId =
            GetKernelDefinitionId(kFindMax1D, kKernelFile, ktt::DimensionVector{ size.n });
          kernelId =
            tuner.CreateSimpleKernel(kFindMax1D + std::to_string(strategyId), definitionId);

          tuner.AddParameter(kernelId, "blockSize", std::vector<uint64_t>{ 32, 64, 128, 256, 512 });

          tuner.AddThreadModifier(kernelId,
            { definitionId },
            ktt::ModifierType::Local,
            ktt::ModifierDimension::X,
            "blockSize",
            ktt::ModifierAction::Multiply);
        }
        return canProcess;
      }

      std::string GetName() const override final { return "Strategy1"; }

      bool Execute(const AExtremaFinder::OutputData &out,
        const AExtremaFinder::InputData &in,
        const Settings &,
        utils::KTTHelper &helper) override final
      {
        if (!in.data.IsValid() || in.data.IsEmpty() || !out.values.IsValid()
            || out.values.IsEmpty())
          return false;

        // prepare input data
        auto &tuner = helper.GetTuner();
        auto argIn = tuner.AddArgumentVector<float>(in.data.ptr,
          in.data.info.GetSize().total,
          ktt::ArgumentAccessType::ReadOnly,
          ktt::ArgumentMemoryLocation::Unified);

        // prepare output data
        auto argVals = tuner.AddArgumentVector<float>(out.values.ptr,
          out.values.info.GetSize().total,
          ktt::ArgumentAccessType::WriteOnly,
          ktt::ArgumentMemoryLocation::Unified);

        auto argSize = tuner.AddArgumentScalar(in.data.info.GetSize().single);

        tuner.SetArguments(definitionId, { argIn, argVals, argSize });

        auto &size = in.data.info.GetSize();
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

    struct Strategy2 : public SingleExtremaFinderCUDA::Strategy
    {
      static constexpr auto kFindMaxRect = "findMaxRect";

      size_t GetHash() const override { return 0; }
      bool IsSimilarTo(const TunableStrategy &ref) const override
      {
        if (GetFullName() != ref.GetFullName()) { return false; }
        // Now we know that type of 'other' is the same as 'this' and we can safely cast it to the
        // needed type
        // auto &o = dynamic_cast<const Strategy2 &>(other);
        // TODO real similarity check
        return false;
      }

      bool Init(const AExtremaFinder::OutputData &,
        const AExtremaFinder::InputData &in,
        const Settings &s,
        utils::KTTHelper &helper) override final
      {
        bool canProcess = (s.version == 1) && (s.location == SearchLocation::kRectCenter)
                          && (s.type == SearchType::kMax) && (s.result == SearchResult::kLocation)
                          && (!in.data.info.IsPadded())
                          && (in.data.dataInfo.type == umpalumpa::data::DataType::kFloat);
        if (canProcess) {
          TunableStrategy::Init(helper);
          auto &size = in.data.info.GetSize();
          auto &tuner = helper.GetTuner();

          // ensure that we have the kernel loaded to KTT
          // this has to be done in critical section, as multiple instances of this algorithm
          // might run on the same worker
          std::lock_guard<std::mutex> lck(helper.GetMutex());
          definitionId = GetKernelDefinitionId(
            kFindMaxRect, kKernelFile, ktt::DimensionVector{ size.n }, { "float" });
          kernelId =
            tuner.CreateSimpleKernel(kFindMaxRect + std::to_string(strategyId), definitionId);

          tuner.AddParameter(
            kernelId, "blockSizeX", std::vector<uint64_t>{ 4, 8, 16, 32, 64, 128 });
          tuner.AddParameter(
            kernelId, "blockSizeY", std::vector<uint64_t>{ 1, 2, 4, 8, 16, 32, 64 });

          tuner.AddConstraint(kernelId,
            { "blockSizeX", "blockSizeY" },
            [&tuner](const std::vector<uint64_t> &params) {
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
        }
        return canProcess;
      }

      std::string GetName() const override final { return "Strategy2"; }

      bool Execute(const AExtremaFinder::OutputData &out,
        const AExtremaFinder::InputData &in,
        const Settings &,
        utils::KTTHelper &helper) override final
      {
        if (!in.data.IsValid() || in.data.IsEmpty() || !out.locations.IsValid()
            || out.locations.IsEmpty())
          return false;

        // prepare input data
        auto &tuner = helper.GetTuner();
        auto argIn = tuner.AddArgumentVector<float>(in.data.ptr,
          in.data.info.GetSize().total,
          ktt::ArgumentAccessType::ReadOnly,
          ktt::ArgumentMemoryLocation::Unified);

        auto argInSize = tuner.AddArgumentScalar(in.data.info.GetSize());

        // prepare output data
        auto argVals = tuner.AddArgumentScalar(NULL);
        auto argLocs = tuner.AddArgumentVector<float>(out.locations.ptr,
          out.values.info.GetSize().total,
          ktt::ArgumentAccessType::WriteOnly,
          ktt::ArgumentMemoryLocation::Unified);

        // FIXME these values should be read from settings
        // FIXME offset + rectDim cant be > inSize, add check
        // Compute the area to search in
        size_t searchRectWidth = 28;
        size_t searchRectHeight = 17;
        size_t searchRectOffsetX = (in.data.info.GetPaddedSize().x - searchRectWidth) / 2;
        size_t searchRectOffsetY = (in.data.info.GetPaddedSize().y - searchRectHeight) / 2;

        auto argOffX = tuner.AddArgumentScalar(searchRectOffsetX);
        auto argOffY = tuner.AddArgumentScalar(searchRectOffsetY);
        auto argRectWidth = tuner.AddArgumentScalar(searchRectWidth);
        auto argRectHeight = tuner.AddArgumentScalar(searchRectHeight);

        tuner.SetArguments(definitionId,
          { argIn, argInSize, argVals, argLocs, argOffX, argOffY, argRectWidth, argRectHeight });

        auto &size = in.data.info.GetSize();
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

  bool SingleExtremaFinderCUDA::Init(const OutputData &out,
    const InputData &in,
    const Settings &settings)
  {
    auto tryToAdd = [this, &out, &in, &settings](auto i) {
      bool canAdd = i->Init(out, in, settings, GetHelper());
      if (canAdd) {
        spdlog::debug("Found valid strategy {}", i->GetName());
        strategy = std::move(i);
      }
      return canAdd;
    };

    return tryToAdd(std::make_unique<Strategy1>()) || tryToAdd(std::make_unique<Strategy2>());
  }

  bool SingleExtremaFinderCUDA::Execute(const OutputData &out,
    const InputData &in,
    const Settings &settings)
  {
    if (!this->IsValid(out, in, settings)) return false;
    return strategy->Execute(out, in, settings, GetHelper());
  }

}// namespace extrema_finder
}// namespace umpalumpa
