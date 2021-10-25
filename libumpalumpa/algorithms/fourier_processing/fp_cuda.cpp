#include <libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/utils/system.hpp>
#include <libumpalumpa/utils/cuda.hpp>

namespace umpalumpa {
namespace fourier_processing {

  namespace {// to avoid poluting
    inline static const auto kKernelFile =
      utils::GetSourceFilePath("libumpalumpa/algorithms/fourier_processing/fp_cuda_kernels.cu");

    struct Strategy1 : public FP_CUDA::Strategy
    {
      // FIXME improve name of the kernel and variable
      static constexpr auto kTMP = "scaleFFT2DKernel";
      // Currently we create one thread per each pixel of a single image. Each thread processes
      // same pixel of all images. The other option for 2D images is to map N dimension to the
      // Z dimension, ie. create more threads, each thread processing fewer images.
      // FIXME  this should be tuned by the KTT

      size_t GetHash() const override { return 0; }
      bool IsSimilar(const TunableStrategy &other) const override
      {
        if (GetFullName() != other.GetFullName()) { return false; }
        // Now we know that type of 'other' is the same as 'this' and we can safely cast it to the
        // needed type
        // auto &o = dynamic_cast<const Strategy1 &>(other);
        // TODO real similarity check
        return false;
      }

      bool Init(const FP_CUDA::OutputData &out,
        const FP_CUDA::InputData &in,
        const Settings &s,
        utils::KTTHelper &helper) override final
      {
        // FIXME check settings
        bool canProcess = (in.data.dataInfo.type == data::DataType::kComplexFloat)
                          && (in.filter.dataInfo.type == data::DataType::kFloat)
                          && (out.data.dataInfo.type == data::DataType::kComplexFloat);

        if (canProcess) {
          TunableStrategy::Init(helper);
          const auto &size = out.data.info.GetPaddedSize();
          auto &tuner = helper.GetTuner();

          // ensure that we have the kernel loaded to KTT
          // this has to be done in critical section, as multiple instances of this algorithm
          // might run on the same worker
          std::lock_guard<std::mutex> lck(helper.GetMutex());
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
        }
        return canProcess;
      }

      std::string GetName() const override final { return "Strategy1"; }

      bool Execute(const FP_CUDA::OutputData &out,
        const FP_CUDA::InputData &in,
        const Settings &s,
        utils::KTTHelper &helper) override final
      {
        if (!in.data.IsValid() || in.data.IsEmpty() || !out.data.IsValid() || out.data.IsEmpty())
          return false;

        auto &tuner = helper.GetTuner();
        // prepare input data
        auto argIn = tuner.AddArgumentVector<float2>(in.data.ptr,
          in.data.info.GetSize().total,
          ktt::ArgumentAccessType::ReadOnly,// FIXME these information should be stored in the
                                            // physical descriptor
          ktt::ArgumentMemoryLocation::Unified);// ^

        // prepare output data
        auto argOut = tuner.AddArgumentVector<float2>(out.data.ptr,
          out.data.info.GetSize().total,
          ktt::ArgumentAccessType::WriteOnly,// FIXME these information should be stored in the
                                             // physical descriptor
          ktt::ArgumentMemoryLocation::Unified);// ^

        auto inSize = tuner.AddArgumentScalar(in.data.info.GetSize());
        auto outSize = tuner.AddArgumentScalar(out.data.info.GetSize());

        auto filter = [&s, &tuner, &in]() {
          if (s.GetApplyFilter()) {
            return tuner.AddArgumentVector<float>(in.filter.ptr,
              in.filter.info.GetSize().total,
              ktt::ArgumentAccessType::ReadOnly,// FIXME these information should be stored in the
                                                // physical descriptor
              ktt::ArgumentMemoryLocation::Unified);// ^
          }
          return tuner.AddArgumentScalar(NULL);
        }();

        // normalize using the original size
        auto normFactor = tuner.AddArgumentScalar(static_cast<float>(in.data.info.GetNormFactor()));

        tuner.SetArguments(definitionId, { argIn, argOut, inSize, outSize, filter, normFactor });

        auto &size = out.data.info.GetPaddedSize();
        tuner.SetLauncher(kernelId, [this, &size](ktt::ComputeInterface &interface) {
          auto blockDim = interface.GetCurrentLocalSize(definitionId);
          ktt::DimensionVector gridDim(size.x, size.y, size.z);
          gridDim.RoundUp(blockDim);
          gridDim.Divide(blockDim);
          interface.RunKernelAsync(definitionId, interface.GetAllQueues().at(0), gridDim, blockDim);
        });

        if (GetTuning()) {
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

  bool FP_CUDA::Init(const OutputData &out, const InputData &in, const Settings &s)
  {
    SetSettings(s);

    auto tryToAdd = [this, &out, &in, &s](auto i) {
      bool canAdd = i->Init(out, in, s, GetHelper());
      if (canAdd) {
        spdlog::debug("Found valid strategy {}", i->GetName());
        strategy = std::move(i);
      }
      return canAdd;
    };

    return tryToAdd(std::make_unique<Strategy1>()) || false;
  }

  bool FP_CUDA::Execute(const OutputData &out, const InputData &in)
  {
    if (!this->IsValid(out, in)) return false;
    return strategy->Execute(out, in, GetSettings(), GetHelper());
  }

}// namespace fourier_processing
}// namespace umpalumpa
