#include <libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/utils/system.hpp>
#include <libumpalumpa/utils/cuda.hpp>

namespace umpalumpa {
namespace fourier_processing {

  namespace {// to avoid poluting

    struct Strategy1 : public FP_CUDA::Strategy
    {
      // Crop, Normalize, Filter, Center
      static constexpr auto kTMP = "scaleFFT2DKernel";
      static constexpr auto kStrategyName = "Strategy1";
      // TODO?? constexpr string concatenation:
      // https://stackoverflow.com/questions/28708497/constexpr-to-concatenate-two-or-more-char-strings
      // TODO NVRTC adds current working directory into the header-search-path.
      // But I would say that absolute path into the project root is better option (makes more sense when including headers in .cu)
      inline static const auto kProjectRoot = utils::GetSourceFilePath(
          "../../..");
      static constexpr auto kCompilerOpts = "--std=c++14 -default-device";
      inline static const auto kKernelFile = utils::GetSourceFilePath(
        "../../../libumpalumpa/algorithms/fourier_processing/fp_cuda_kernels.cu");
      // FIXME how to set/tune this via KTT (filip)
      static constexpr auto kBlockDimX = 32;
      static constexpr auto kBlockDimY = 32;
      // Currently we create one thread per each pixel of a single image. Each thread processes
      // same pixel of all images. The other option for 2D images is to map N dimension to the
      // Z dimension, ie. create more threads, each thread processing fewer images.
      // FIXME  this should be tuned by the KTT
      static constexpr auto kBlockDimZ = 1;

      KernelData kernelData;

      inline size_t ComputeDimension(size_t l, int r) const {
        return static_cast<size_t>(std::ceil(
              static_cast<float>(l) / static_cast<float>(r)));
      }

      bool Init(const FP_CUDA::OutputData &out,
        const FP_CUDA::InputData &,
        const Settings &s,
        ktt::Tuner &tuner) override final
      {
        bool canProcess = true;// FIXME 

        if (canProcess) {
          const ktt::DimensionVector blockDimensions(kBlockDimX, kBlockDimY, kBlockDimZ);
          const auto &size = out.data.info.GetPaddedSize();
          const ktt::DimensionVector gridDimensions(
              ComputeDimension(size.x, kBlockDimX),
              ComputeDimension(size.y, kBlockDimY),
              ComputeDimension(size.z, kBlockDimZ));

          kernelData.definitionIds.emplace_back(tuner.AddKernelDefinitionFromFile(
            kTMP, kKernelFile, gridDimensions, blockDimensions, {}));
          kernelData.kernelId = tuner.CreateSimpleKernel(kTMP, kernelData.definitionIds.front());
          tuner.AddParameter(kernelData.kernelId, "applyFilter", std::vector<uint64_t>{ s.GetApplyFilter() });
          tuner.AddParameter(kernelData.kernelId, "normalize", std::vector<uint64_t>{ s.GetNormalize() });
          tuner.AddParameter(kernelData.kernelId, "center", std::vector<uint64_t>{ s.GetCenter() });
          tuner.SetCompilerOptions("-I" + kProjectRoot + " " + kCompilerOpts);
        }
        return canProcess;
      }

      std::string GetName() const override final { return kStrategyName; }

      bool Execute(const FP_CUDA::OutputData &out,
        const FP_CUDA::InputData &in,
        const Settings &s,
        ktt::Tuner &tuner) override final
      {
        if (!in.data.IsValid() || in.data.IsEmpty() || !out.data.IsValid()
            || out.data.IsEmpty())
          return false;

        // prepare input data
        auto argIn = tuner.AddArgumentVector<float2>(in.data.ptr,
          in.data.info.GetSize().total,
          ktt::ArgumentAccessType::ReadOnly,// FIXME these information should be stored in the physical descriptor
          ktt::ArgumentMemoryLocation::Unified);// ^

        // prepare output data
        auto argOut = tuner.AddArgumentVector<float2>(out.data.ptr,
          out.data.info.GetSize().total,
          ktt::ArgumentAccessType::WriteOnly,// FIXME these information should be stored in the physical descriptor
          ktt::ArgumentMemoryLocation::Unified);// ^

        //TODO add filter
        //auto filter = tuner.AddArgumentVector<float>(out.data.ptr,
        //  out.data.info.GetSize().total,
        //  ktt::ArgumentAccessType::WriteOnly,// FIXME these information should be stored in the physical descriptor
        //  ktt::ArgumentMemoryLocation::Unified);// ^

        //auto noOfImages = tuner.AddArgumentScalar(in.data.info.GetSize().n);
        //auto inX = tuner.AddArgumentScalar(in.data.info.GetSize().x);
        //auto inY = tuner.AddArgumentScalar(in.data.info.GetSize().y);
        auto inSize = tuner.AddArgumentScalar(in.data.info.GetSize());
        //auto outX = tuner.AddArgumentScalar(out.data.info.GetSize().x);
        //auto outY = tuner.AddArgumentScalar(out.data.info.GetSize().y);
        auto outSize = tuner.AddArgumentScalar(out.data.info.GetSize());
        auto filter = tuner.AddArgumentScalar(size_t(0));// instead of nullptr, KTT is not happy with pointers in scalar arguments
        auto normFactor = tuner.AddArgumentScalar(1.f / static_cast<float>(in.data.info.GetSize().single));

        tuner.SetArguments(kernelData.definitionIds.front(),
            { argIn, argOut, inSize, outSize, filter, normFactor });
            //{ argIn, argOut, noOfImages, inX, inY, outX, outY, filter, normFactor });

        // update grid dimension to properly react to batch size
        tuner.SetLauncher(kernelData.kernelId, [this, &out](ktt::ComputeInterface &interface) {
          const ktt::DimensionVector blockDimensions(kBlockDimX, kBlockDimY, kBlockDimZ);
          const auto &size = out.data.info.GetPaddedSize();
          const ktt::DimensionVector gridDimensions(
              ComputeDimension(size.x, kBlockDimX),
              ComputeDimension(size.y, kBlockDimY),
              ComputeDimension(size.z, kBlockDimZ));
          interface.RunKernelAsync(kernelData.definitionIds.front(),
            interface.GetAllQueues().at(0),
            gridDimensions,
            blockDimensions);
        });

        auto configuration =
          tuner.CreateConfiguration(kernelData.kernelId, {
              { "applyFilter", static_cast<uint64_t>(s.GetApplyFilter()) },
              { "normalize", static_cast<uint64_t>(s.GetNormalize()) },
              { "center", static_cast<uint64_t>(s.GetCenter()) } });
        tuner.Run(kernelData.kernelId, configuration, {}); // run is blocking call
        // arguments shall be removed once the run is done
        return true;
      };
    };
  }// namespace

  void FP_CUDA::Synchronize() { tuner.Synchronize(); }

  // FIXME createApiInitializer is reapeating itself in all gpu implementation, find a way to implement it once
  ktt::ComputeApiInitializer FP_CUDA::createApiInitializer(int deviceOrdinal)
  {
    CudaErrchk(cuInit(0));
    CUdevice device;
    CudaErrchk(cuDeviceGet(&device, deviceOrdinal));
    CUcontext context;
    cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
    CUstream stream;
    CudaErrchk(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    return ktt::ComputeApiInitializer(context, std::vector<ktt::ComputeQueue>{ stream });
  }

  ktt::ComputeApiInitializer FP_CUDA::createApiInitializer(CUstream stream)
  {
    CudaErrchk(cuInit(0));
    CUcontext context;
    CudaErrchk(cuStreamGetCtx(stream, &context));
    // Create compute API initializer which specifies context and streams that will be utilized by
    // the tuner.
    return ktt::ComputeApiInitializer(context, std::vector<ktt::ComputeQueue>{ stream });
  }


  bool FP_CUDA::Init(const OutputData &out,
    const InputData &in,
    const Settings &s)
  {
    SetSettings(s);

    auto tryToAdd = [this, &out, &in, &s](auto i) {
      bool canAdd = i->Init(out, in, s, tuner);
      if (canAdd) {
        spdlog::debug("Found valid strategy {}", i->GetName());
        strategy = std::move(i);
      }
      return canAdd;
    };

    return tryToAdd(std::make_unique<Strategy1>()) || false;
  }

  bool FP_CUDA::Execute(const OutputData &out,
    const InputData &in)
  {
    if (!this->IsValid(out, in)) return false;
    return strategy->Execute(out, in, GetSettings(), tuner);
  }

}// namespace fourier_processing 
}// namespace umpalumpa
 
