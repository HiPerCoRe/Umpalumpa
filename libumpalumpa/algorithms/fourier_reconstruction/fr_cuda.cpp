#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cuda.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space.hpp>

namespace umpalumpa::fourier_reconstruction {

namespace {// to avoid poluting
  inline static const auto kKernelFile =
    utils::GetSourceFilePath("libumpalumpa/algorithms/fourier_reconstruction/fr_cuda_kernels.cu");

  struct Strategy1 final : public FRCUDA::KTTStrategy
  {
    // Inherit constructor
    using FRCUDA::KTTStrategy::KTTStrategy;


    // FIXME improve name of the kernel and variable
    static constexpr auto kTMP = "ProcessKernel";

    size_t GetHash() const override { return 0; }
    bool IsSimilarTo(const TunableStrategy &) const override
    {
      // TODO real similarity check
      return false;
    }

    bool InitImpl() override
    {
      const auto &in = alg.Get().GetInputRef();
      // we can process only odd sized data
      if (0 == in.GetVolume().info.GetSize().x % 2) { return false; }

      // FIXME we have to synchronize to make sure that the constants are not being used ATM
      constants = AFR::CreateConstants(in, alg.Get().GetSettings());

      const auto &size = in.GetVolume().info.GetSize();
      auto &tuner = kttHelper.GetTuner();

      // ensure that we have the kernel loaded to KTT
      // this has to be done in critical section, as multiple instances of this algorithm
      // might run on the same worker
      const auto &s = alg.Get().GetSettings();
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
      tuner.AddParameter(kernelId, "BLOCK_DIM", std::vector<uint64_t>{ 8, 16 });
      tuner.AddParameter(kernelId, "SHARED_BLOB_TABLE", std::vector<uint64_t>{ 0, 1 });
      tuner.AddParameter(kernelId, "SHARED_IMG", std::vector<uint64_t>{ 0, 1 });
      tuner.AddParameter(kernelId,
        "PRECOMPUTE_BLOB_VAL",
        std::vector<uint64_t>{ s.GetInterpolation() == Settings::Interpolation::kLookup });
      tuner.AddParameter(kernelId, "TILE", std::vector<uint64_t>{ 2, 4, 8 });
      tuner.AddParameter(kernelId, "GRID_DIM_Z", std::vector<uint64_t>{ 1, 8 });
      tuner.AddParameter(kernelId,
        "BLOB_TABLE_SIZE_SQRT",
        std::vector<uint64_t>{ in.GetBlobTable().info.GetSize().total });

      // FIXME add constraints, probably from the old branch or the paper
      tuner.AddConstraint(
        kernelId, { "BLOCK_DIM", "BLOCK_DIM" }, [&tuner](const std::vector<uint64_t> &params) {
          return params[0] * params[1] <= tuner.GetCurrentDeviceInfo().GetMaxWorkGroupSize();
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

      // divide gridsize in specified dimension by 'BLOCK_DIM' so that it's multiple of blocksize
      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Global,
        ktt::ModifierDimension::X,
        "BLOCK_DIM",
        ktt::ModifierAction::DivideCeil);
      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Global,
        ktt::ModifierDimension::Y,
        "BLOCK_DIM",
        ktt::ModifierAction::DivideCeil);

      tuner.SetSearcher(kernelId, std::make_unique<ktt::RandomSearcher>());
      return true;
    }

    std::string GetName() const override { return "Strategy1"; }

    bool Execute(const FRCUDA::OutputData &, const FRCUDA::InputData &in) override
    {
      if (!in.GetVolume().IsValid() || in.GetVolume().IsEmpty() || !in.GetFFT().IsValid()
          || in.GetFFT().IsEmpty() || !in.GetWeight().IsValid() || in.GetWeight().IsEmpty())
        return false;

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
      auto argConstants = tuner.AddArgumentScalar(constants);

      auto argSize = tuner.AddArgumentScalar(in.GetFFT().info.GetSize());
      auto argSpaceCount = tuner.AddArgumentScalar(in.GetTraverseSpace().info.GetSize());

      auto argSharedMemSize = tuner.AddArgumentLocal<uint64_t>(1); //must be non-zero value because KTT

      auto argConstConstants =
        tuner.AddArgumentSymbol<Constants>(AFR::CreateConstants(in, alg.GetSettings()), "constants");


      auto definitionId = GetDefinitionId();
      auto kernelId = GetKernelId();

      tuner.SetArguments(definitionId,
        { argVolume,
          argWeight,
          argSize,
          argSpaceCount,
          argSpace,
          argFFT,
          argTable,
          argConstants,
          argSharedMemSize,
          argConstConstants 
          });

      auto &size = in.GetVolume().info.GetSize();
      tuner.SetLauncher(
        kernelId, [definitionId, &size, &argSharedMemSize](ktt::ComputeInterface &interface) {
          auto blockDim = interface.GetCurrentLocalSize(definitionId);
          // FIXME check sizes
          ktt::DimensionVector gridDim(size.x, size.y, size.z);
          gridDim.RoundUp(blockDim);
          gridDim.Divide(blockDim);
          auto &pairs = interface.GetCurrentConfiguration().GetPairs();
          bool useSharedImg = ktt::ParameterPair::GetParameterValue<uint64_t>(pairs, "SHARED_IMG");
          if (!useSharedImg) {
            // const int imgCacheDim = static_cast<int>(ceil(sqrt(2.f) * sqrt(3.f) * (BLOCK_DIM + 2
            // * blobRadius))); SHARED_IMG ? (imgCacheDim*imgCacheDim*sizeof(float2)) : 0;
            // size_t dataSize = 0;
            // interface.UpdateLocalArgument(argSharedMemSize, dataSize);
          }
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
          { { "BLOCK_DIM", static_cast<uint64_t>(16) },
            { "SHARED_IMG", static_cast<uint64_t>(0) },
            { "TILE", static_cast<uint64_t>(2) },
            { "GRID_DIM_Z", static_cast<uint64_t>(1) } });
        tuner.Run(kernelId, bestConfig, {});// run is blocking call
        // arguments shall be removed once the run is done
      }

      return true;
    };

  private:
    Constants constants;
  };
}// namespace

void FRCUDA::Synchronize() { GetHelper().GetTuner().Synchronize(); }

std::vector<std::unique_ptr<FRCUDA::Strategy>> FRCUDA::GetStrategies() const
{
  std::vector<std::unique_ptr<FRCUDA::Strategy>> vec;
  vec.emplace_back(std::make_unique<Strategy1>(*this));
  return vec;
}


}// namespace umpalumpa::fourier_reconstruction