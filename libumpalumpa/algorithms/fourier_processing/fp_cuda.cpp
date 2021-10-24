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
      static constexpr auto kBlockDimX = 32;
      static constexpr auto kBlockDimY = 32;
      static constexpr auto kBlockDimZ = 1;

      inline size_t ComputeDimension(size_t l, int r) const
      {
        return static_cast<size_t>(std::ceil(static_cast<float>(l) / static_cast<float>(r)));
      }

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
          const ktt::DimensionVector blockDimensions(kBlockDimX, kBlockDimY, kBlockDimZ);
          const auto &size = out.data.info.GetPaddedSize();
          const ktt::DimensionVector gridDimensions(ComputeDimension(size.x, kBlockDimX),
            ComputeDimension(size.y, kBlockDimY),
            ComputeDimension(size.z, kBlockDimZ));

          auto &tuner = helper.GetTuner();
          // ensure that we have the kernel loaded to KTT
          // this has to be done in critical section, as multiple instances of this algorithm
          // might run on the same worker
          std::lock_guard<std::mutex> lck(helper.GetMutex());
          definitionId = tuner.GetKernelDefinitionId(kTMP);
          if (definitionId == ktt::InvalidKernelDefinitionId) {
            definitionId = tuner.AddKernelDefinitionFromFile(
              kTMP, kKernelFile, gridDimensions, blockDimensions, {});
          }
          kernelId = tuner.CreateSimpleKernel(kTMP + std::to_string(strategyId), definitionId);
          tuner.AddParameter(kernelId, "applyFilter", std::vector<uint64_t>{ s.GetApplyFilter() });
          tuner.AddParameter(kernelId, "normalize", std::vector<uint64_t>{ s.GetNormalize() });
          tuner.AddParameter(kernelId, "center", std::vector<uint64_t>{ s.GetCenter() });
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

        // update grid dimension to properly react to batch size
        tuner.SetLauncher(
          kernelId, [this, &out, definitionId = definitionId](ktt::ComputeInterface &interface) {
            const ktt::DimensionVector blockDimensions(kBlockDimX, kBlockDimY, kBlockDimZ);
            const auto &size = out.data.info.GetPaddedSize();
            const ktt::DimensionVector gridDimensions(ComputeDimension(size.x, kBlockDimX),
              ComputeDimension(size.y, kBlockDimY),
              ComputeDimension(size.z, kBlockDimZ));
            interface.RunKernelAsync(
              definitionId, interface.GetAllQueues().at(0), gridDimensions, blockDimensions);
          });

        auto configuration = tuner.CreateConfiguration(kernelId,
          { { "applyFilter", static_cast<uint64_t>(s.GetApplyFilter()) },
            { "normalize", static_cast<uint64_t>(s.GetNormalize()) },
            { "center", static_cast<uint64_t>(s.GetCenter()) } });
        tuner.Run(kernelId, configuration, {});// run is blocking call
        // FIXME arguments shall be removed once the run is done
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
