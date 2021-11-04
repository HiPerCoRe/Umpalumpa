#include <libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp>
#include <libumpalumpa/tuning/tunable_strategy.hpp>

namespace umpalumpa::fourier_processing {

namespace {// to avoid poluting
  inline static const auto kKernelFile =
    utils::GetSourceFilePath("libumpalumpa/algorithms/fourier_processing/fp_cuda_kernels.cu");

  struct Strategy1 final : public FP_CUDA::KTTStrategy
  {
    // Inherit constructor
    using FP_CUDA::KTTStrategy::KTTStrategy;

    // FIXME improve name of the kernel and variable
    static constexpr auto kTMP = "scaleFFT2DKernel";
    // Currently we create one thread per each pixel of a single image. Each thread processes
    // same pixel of all images. The other option for 2D images is to map N dimension to the
    // Z dimension, ie. create more threads, each thread processing fewer images.
    // FIXME  this should be tuned by the KTT

    size_t GetHash() const override { return 0; }
    bool IsSimilarTo(const TunableStrategy &) const override
    {
      // auto &o = dynamic_cast<const Strategy1 &>(other);
      // TODO real similarity check
      return false;
    }

    bool InitImpl() override
    {
      // FIXME check settings
      const auto &out = alg.Get().GetOutputRef();
      const auto &in = alg.Get().GetInputRef();
      if (!AFP::IsFloat(out, in)) return false;

      const auto &size = out.GetData().info.GetPaddedSize();
      auto &tuner = kttHelper.GetTuner();

      // ensure that we have the kernel loaded to KTT
      // this has to be done in critical section, as multiple instances of this algorithm
      // might run on the same worker
      const auto &s = alg.Get().GetSettings();
      std::lock_guard<std::mutex> lck(kttHelper.GetMutex());
      definitionId = GetKernelDefinitionId(kTMP,
        kKernelFile,
        ktt::DimensionVector{ size.x, size.y, size.z },
        { std::to_string(s.GetApplyFilter()),
          std::to_string(s.GetNormalize()),
          std::to_string(s.GetCenter()) });
      kernelId = tuner.CreateSimpleKernel(kTMP + std::to_string(strategyId), definitionId);

      tuner.AddParameter(
        kernelId, "blockSizeX", std::vector<uint64_t>{ 1, 2, 4, 8, 16, 32, 64, 128 });
      tuner.AddParameter(
        kernelId, "blockSizeY", std::vector<uint64_t>{ 1, 2, 4, 8, 16, 32, 64, 128 });

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

      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Global,
        ktt::ModifierDimension::X,
        "blockSizeX",
        ktt::ModifierAction::DivideCeil);
      tuner.AddThreadModifier(kernelId,
        { definitionId },
        ktt::ModifierType::Global,
        ktt::ModifierDimension::Y,
        "blockSizeY",
        ktt::ModifierAction::DivideCeil);

      tuner.SetSearcher(kernelId, std::make_unique<ktt::RandomSearcher>());
      return true;
    }

    std::string GetName() const override { return "Strategy1"; }

    bool Execute(const FP_CUDA::OutputData &out, const FP_CUDA::InputData &in) override
    {
      if (!in.GetData().IsValid() || in.GetData().IsEmpty() || !out.GetData().IsValid()
          || out.GetData().IsEmpty())
        return false;

      auto &tuner = kttHelper.GetTuner();
      // prepare input data
      auto argIn = tuner.AddArgumentVector<float2>(in.GetData().ptr,
        in.GetData().info.GetSize().total,
        ktt::ArgumentAccessType::ReadOnly,// FIXME these information should be stored in the
                                          // physical descriptor
        ktt::ArgumentMemoryLocation::Unified);// ^

      // prepare output data
      auto argOut = tuner.AddArgumentVector<float2>(out.GetData().ptr,
        out.GetData().info.GetSize().total,
        ktt::ArgumentAccessType::WriteOnly,// FIXME these information should be stored in the
                                           // physical descriptor
        ktt::ArgumentMemoryLocation::Unified);// ^

      auto inSize = tuner.AddArgumentScalar(in.GetData().info.GetSize());
      auto outSize = tuner.AddArgumentScalar(out.GetData().info.GetSize());

      const auto &s = alg.Get().GetSettings();
      auto filter = [&s, &tuner, &in]() {
        if (s.GetApplyFilter()) {
          return tuner.AddArgumentVector<float>(in.GetFilter().ptr,
            in.GetFilter().info.GetSize().total,
            ktt::ArgumentAccessType::ReadOnly,// FIXME these information should be stored in the
                                              // physical descriptor
            ktt::ArgumentMemoryLocation::Unified);// ^
        }
        return tuner.AddArgumentScalar(NULL);
      }();

      // normalize using the original size
      auto normFactor =
        tuner.AddArgumentScalar(static_cast<float>(in.GetData().info.GetNormFactor()));

      tuner.SetArguments(definitionId, { argIn, argOut, inSize, outSize, filter, normFactor });

      auto &size = out.GetData().info.GetPaddedSize();
      tuner.SetLauncher(kernelId, [this, &size](ktt::ComputeInterface &interface) {
        auto blockDim = interface.GetCurrentLocalSize(definitionId);
        ktt::DimensionVector gridDim(size.x, size.y, size.z);
        gridDim.RoundUp(blockDim);
        gridDim.Divide(blockDim);
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
          { { "blockSizeX", static_cast<uint64_t>(32) },
            { "blockSizeY", static_cast<uint64_t>(8) } });
        tuner.Run(kernelId, bestConfig, {});// run is blocking call
        // arguments shall be removed once the run is done
      }

      return true;
    };
  };
}// namespace

void FP_CUDA::Synchronize() { GetHelper().GetTuner().Synchronize(); }

std::vector<std::unique_ptr<FP_CUDA::Strategy>> FP_CUDA::GetStrategies() const
{
  std::vector<std::unique_ptr<FP_CUDA::Strategy>> vec;
  vec.emplace_back(std::make_unique<Strategy1>(*this));
  return vec;
}

}// namespace umpalumpa::fourier_processing
