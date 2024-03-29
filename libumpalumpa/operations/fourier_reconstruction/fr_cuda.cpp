#include <libumpalumpa/operations/fourier_reconstruction/fr_cuda.hpp>
#include <libumpalumpa/operations/fourier_reconstruction/traverse_space.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>
#include <libumpalumpa/tuning/strategy_group.hpp>

namespace umpalumpa::fourier_reconstruction {

namespace {// to avoid poluting
  inline static const auto kKernelFile =
    utils::GetSourceFilePath("libumpalumpa/operations/fourier_reconstruction/fr_cuda_kernels.cu");

  struct Strategy1 : public FRCUDA::KTTStrategy
  {
    // Inherit constructor
    using FRCUDA::KTTStrategy::KTTStrategy;


    // FIXME improve name of the kernel and variable
    static constexpr auto kTMP = "ProcessKernel";

    size_t GetHash() const override { return 0; }

    std::unique_ptr<tuning::Leader> CreateLeader() const override
    {
      return tuning::StrategyGroup::CreateLeader(*this, op);
    }
    tuning::StrategyGroup LoadTuningData() const override
    {
      return tuning::StrategyGroup::LoadTuningData(*this, op);
    }

    std::vector<ktt::KernelConfiguration> GetDefaultConfigurations() const override
    {
      return { kttHelper.GetTuner().CreateConfiguration(GetKernelId(),
        { { "BLOCK_DIM", static_cast<uint64_t>(16) },
          { "SHARED_IMG", static_cast<uint64_t>(0) },
          { "TILE", static_cast<uint64_t>(2) },
          { "GRID_DIM_Z", static_cast<uint64_t>(1) } }) };
    }

    bool IsEqualTo(const TunableStrategy &ref) const override
    {
      bool isEqual = true;
      try {
        auto &refStrat = dynamic_cast<const Strategy1 &>(ref);
        isEqual = isEqual && GetOutputRef().IsEquivalentTo(refStrat.GetOutputRef());
        isEqual = isEqual && GetInputRef().IsEquivalentTo(refStrat.GetInputRef());
        isEqual = isEqual && GetSettings().IsEquivalentTo(refStrat.GetSettings());
        // Additional checks might be needed
      } catch (std::bad_cast &) {
        isEqual = false;
      }
      return isEqual;
    }

    bool IsSimilarTo(const TunableStrategy &ref) const override
    {
      bool isSimilar = true;
      try {
        auto &refStrat = dynamic_cast<const Strategy1 &>(ref);
        isSimilar = isSimilar && GetOutputRef().IsEquivalentTo(refStrat.GetOutputRef());
        isSimilar = isSimilar && GetInputRef().IsEquivalentTo(refStrat.GetInputRef());
        isSimilar = isSimilar && GetSettings().IsEquivalentTo(refStrat.GetSettings());
        // Using naive similarity: same as equality
      } catch (std::bad_cast &) {
        isSimilar = false;
      }
      return isSimilar;
    }

    bool InitImpl() override
    {
      const auto &in = op.Get().GetInputRef();
      // we can process only odd sized data
      if (0 == in.GetVolume().info.GetSize().x % 2) { return false; }

      // FIXME we have to synchronize to make sure that the constants are not being used ATM
      constants = AFR::CreateConstants(in, op.Get().GetSettings());

      const auto &size = in.GetVolume().info.GetSize();
      auto &tuner = kttHelper.GetTuner();

      // ensure that we have the kernel loaded to KTT
      // this has to be done in critical section, as multiple instances of this operation
      // might run on the same worker
      const auto &s = op.Get().GetSettings();
      std::lock_guard<std::mutex> lck(kttHelper.GetMutex());
      AddKernelDefinition(kTMP,
        kKernelFile,
        ktt::DimensionVector{ size.x, size.y, size.z },
        { std::to_string(ToInt(s.GetBlobOrder())),
          std::to_string(s.GetType() == Settings::Type::kFast),
          std::to_string(s.GetAlpha() <= 15.f) });
      auto definitionId = GetDefinitionId();

      AddKernel(kTMP, definitionId);
      auto kernelId = GetKernelId();

      // FIXME add more values, probably from the old branch or the paper
      tuner.AddParameter(kernelId, "BLOCK_DIM", std::vector<uint64_t>{ 8, 12, 16, 20, 24, 28, 32 });
      tuner.AddParameter(kernelId, "SHARED_BLOB_TABLE", std::vector<uint64_t>{ 0, 1 });
      tuner.AddParameter(kernelId, "SHARED_IMG", std::vector<uint64_t>{ 0, 1 });
      tuner.AddParameter(kernelId,
        "PRECOMPUTE_BLOB_VAL",
        std::vector<uint64_t>{ s.GetInterpolation() == Settings::Interpolation::kLookup });
      tuner.AddParameter(kernelId, "TILE", std::vector<uint64_t>{ 1, 2, 4, 8 });
      tuner.AddParameter(kernelId, "GRID_DIM_Z", std::vector<uint64_t>{ 1, 2, 4, 8, 16 });
      tuner.AddParameter(kernelId,
        "BLOB_TABLE_SIZE_SQRT",
        std::vector<uint64_t>{ in.GetBlobTable().info.GetSize().total });

      // FIXME add constraints, probably from the old branch or the paper
      tuner.AddConstraint(
        kernelId, { "BLOCK_DIM", "BLOCK_DIM" }, [&tuner](const std::vector<uint64_t> &params) {
          return params[0] * params[1] <= tuner.GetCurrentDeviceInfo().GetMaxWorkGroupSize();
        });

      tuner.AddConstraint(
        kernelId, { "BLOCK_DIM", "TILE" }, [](const std::vector<uint64_t> &params) {
          return params.at(1) == 1 || (params.at(0) % params.at(1) == 0);
        });

      tuner.AddConstraint(kernelId,
        { "SHARED_IMG", "TILE" },
        [](const std::vector<uint64_t> &params) { return params.at(0) == 0 || params.at(1) == 1; });

      tuner.AddConstraint(kernelId,
        { "BLOCK_DIM", "TILE" },
        [](const std::vector<uint64_t> &params) { return params.at(0) > params.at(1); });

      // TODO if / when KTT supports reporting of the used shared memory, we can enhance this check
      tuner.AddConstraint(
        kernelId, { "SHARED_BLOB_TABLE", "SHARED_IMG" }, [](const std::vector<uint64_t> &params) {
          return !(params.at(0) == 1 && params.at(1) == 1);
        });

      tuner.AddConstraint(kernelId,
        { "SHARED_BLOB_TABLE", "PRECOMPUTE_BLOB_VAL" },
        [](const std::vector<uint64_t> &params) {
          return params.at(0) == 0 || (params.at(0) == 1 && params.at(1) == 1);
        });

      // multiply blocksize in specified dimension by 'BLOCK_DIM'
      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Local,
        ktt::ModifierDimension::X,
        "BLOCK_DIM",
        ktt::ModifierAction::Multiply);
      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Local,
        ktt::ModifierDimension::Y,
        "BLOCK_DIM",
        ktt::ModifierAction::Multiply);

      tuner.SetSearcher(kernelId, std::make_unique<ktt::RandomSearcher>());
      return true;
    }

    std::string GetName() const override { return "Strategy1"; }

    bool Execute(const FRCUDA::OutputData &, const FRCUDA::InputData &in) override
    {
      if (!in.GetVolume().IsValid() || in.GetVolume().IsEmpty() || !in.GetFFT().IsValid()
          || in.GetFFT().IsEmpty() || !in.GetWeight().IsValid() || in.GetWeight().IsEmpty())
        return false;

      const auto &s = op.GetSettings();

      auto &tuner = kttHelper.GetTuner();
      // TODO add type check
      auto argVolume =
        AddArgumentVector<float2>(in.GetVolume(), ktt::ArgumentAccessType::ReadWrite);
      auto argWeight = AddArgumentVector<float>(in.GetWeight(), ktt::ArgumentAccessType::ReadWrite);
      auto argFFT = AddArgumentVector<float2>(in.GetFFT(), ktt::ArgumentAccessType::ReadOnly);
      auto argSpace =
        AddArgumentVector<TraverseSpace>(in.GetTraverseSpace(), ktt::ArgumentAccessType::ReadOnly);
      auto argTable =
        AddArgumentVector<float>(in.GetBlobTable(), ktt::ArgumentAccessType::ReadOnly);
      auto argImgCacheDim = tuner.AddArgumentScalar(0);// value will be replaced in the launcher

      auto argSize = tuner.AddArgumentScalar(in.GetFFT().info.GetSize());
      auto argSpaceCount = tuner.AddArgumentScalar(in.GetTraverseSpace().info.GetSize().n);

      // FIXME change to 0 once KTT supports it
      auto argSharedMemSize =
        tuner.AddArgumentLocal<uint64_t>(1);// must be non-zero value because KTT

      auto argConstants =
        tuner.AddArgumentSymbol<Constants>(AFR::CreateConstants(in, s),"", "constants");

      auto definitionId = GetDefinitionId();
      auto kernelId = GetKernelId();

      SetArguments(definitionId,
        { argVolume,
          argWeight,
          argSize,
          argSpaceCount,
          argSpace,
          argFFT,
          argTable,
          argImgCacheDim,
          argSharedMemSize,
          argConstants });

      auto &size = in.GetVolume().info.GetSize();
      tuner.SetLauncher(kernelId,
        [this, &size, &argSharedMemSize, &argImgCacheDim, &s](ktt::ComputeInterface &interface) {
          auto &pairs = interface.GetCurrentConfiguration().GetPairs();
          auto blockDim = interface.GetCurrentLocalSize(GetDefinitionId());
          ktt::DimensionVector gridDim(size.x, size.y);
          gridDim.RoundUp(blockDim);
          gridDim.Divide(blockDim);
          gridDim.SetSizeZ(ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "GRID_DIM_Z"));
          bool useSharedImg = ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "SHARED_IMG");
          if (useSharedImg) {
            auto imgCacheDim = size_t(
              ceil(sqrt(2.f) * sqrt(3.f) * (float(blockDim.GetSizeX()) + 2 * s.GetBlobRadius())));
            size_t dataSize = imgCacheDim * imgCacheDim * sizeof(float2);
            interface.UpdateScalarArgument(argImgCacheDim, &imgCacheDim);
            interface.UpdateLocalArgument(argSharedMemSize, dataSize);
          }
          if (ShouldBeTuned(GetKernelId())) {
            interface.RunKernel(GetKernelId(), gridDim, blockDim);
          } else {
            WaitBeforeDestruction(interface.RunKernelAsync(
              GetDefinitionId(), interface.GetAllQueues().at(0), gridDim, blockDim));
          }
        });

      ExecuteKernel(kernelId);

      return true;
    };

  private:
    Constants constants;
  };// namespace
}// namespace

void FRCUDA::Synchronize() { GetHelper().GetTuner().SynchronizeDevice(); }

std::vector<std::unique_ptr<FRCUDA::Strategy>> FRCUDA::GetStrategies() const
{
  std::vector<std::unique_ptr<FRCUDA::Strategy>> vec;
  vec.emplace_back(std::make_unique<Strategy1>(*this));
  return vec;
}


}// namespace umpalumpa::fourier_reconstruction
