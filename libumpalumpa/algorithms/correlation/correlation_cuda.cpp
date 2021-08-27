#include <libumpalumpa/algorithms/correlation/correlation_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/utils/system.hpp>
#include <libumpalumpa/utils/cuda.hpp>

namespace umpalumpa {
namespace correlation {

  namespace {// to avoid poluting

    struct Strategy1 : public Correlation_CUDA::Strategy
    {
      // Crop, Normalize, Filter, Center
      static constexpr auto kTMP = "scaleFFT2DKernel";
      static constexpr auto kStrategyName = "Strategy1";
      // c++ is sometimes difficult :(
      // TODO?? constexpr string concatenation:
      // https://stackoverflow.com/questions/28708497/constexpr-to-concatenate-two-or-more-char-strings
      //static constexpr auto kProjectRoot = "../../..";
      static constexpr auto kIncludePath = "-I../../..";
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

      static constexpr auto kTile = 8;

      KernelData kernelData;

      inline size_t ComputeDimension(size_t l, int r) const {
        return static_cast<size_t>(std::ceil(
              static_cast<float>(l) / static_cast<float>(r)));
      }

      bool Init(const Correlation_CUDA::OutputData &out,
        const Correlation_CUDA::InputData &,
        const Settings &s,
        ktt::Tuner &tuner) override final
      {
        bool canProcess = true;// FIXME 

        if (canProcess) {
          const ktt::DimensionVector blockDimensions(kBlockDimX * kBlockDimY * kBlockDimZ);
          const auto &size = out.data.info.GetPaddedSize();
          const ktt::DimensionVector gridDimensions(
              ComputeDimension(size.x, kBlockDimX),
              ComputeDimension(size.y, kBlockDimY),
              ComputeDimension(size.z, kBlockDimZ));

          kernelData.definitionIds.emplace_back(tuner.AddKernelDefinitionFromFile(
            kTMP, kKernelFile, gridDimensions, blockDimensions, {}));
          kernelData.kernelId = tuner.CreateSimpleKernel(kTMP, kernelData.definitionIds.front());
          tuner.AddParameter(kernelData.kernelId, "center", std::vector<uint64_t>{ s.GetCenter() });
          tuner.SetCompilerOptions(kIncludePath);
        }
        return canProcess;
      }

      std::string GetName() const override final { return kStrategyName; }

      bool Execute(const Correlation_CUDA::OutputData &out,
        const Correlation_CUDA::InputData &in,
        const Settings &s,
        ktt::Tuner &tuner) override final
      {
        if (!in.data1.IsValid() || in.data1.IsEmpty() || !out.data.IsValid()//FIXME refactor
            || out.data.IsEmpty())
          return false;

//void correlate2D(T* __restrict__ correlations, const T* __restrict__ in1, umpalumpa::data::Size in1Size,
//    const T* __restrict__ in2, int in2N) {

        // prepare input data1
        auto argIn1 = tuner.AddArgumentVector<float>(in.data1.data,
          in.data1.info.GetSize().total,
          ktt::ArgumentAccessType::ReadOnly,// FIXME these information should be stored in the physical descriptor
          ktt::ArgumentMemoryLocation::Unified);// ^

        auto argIn2 = tuner.AddArgumentVector<float>(in.data2.data,
          in.data2.info.GetSize().total,
          ktt::ArgumentAccessType::ReadOnly,// FIXME these information should be stored in the physical descriptor
          ktt::ArgumentMemoryLocation::Unified);// ^

        // prepare output data1
        auto argOut = tuner.AddArgumentVector<float>(out.data.data,
          out.data.info.GetSize().total,
          ktt::ArgumentAccessType::WriteOnly,// FIXME these information should be stored in the physical descriptor
          ktt::ArgumentMemoryLocation::Unified);// ^

        auto inSize = tuner.AddArgumentScalar(in.data1.info.GetSize());
        auto in2N = tuner.AddArgumentScalar(in.data2.info.GetSize().n);

        tuner.SetArguments(kernelData.definitionIds.front(),
            { argOut, argIn1,  inSize, argIn2, in2N });

        // update grid dimension to properly react to batch size
        tuner.SetLauncher(kernelData.kernelId, [this, &out](ktt::ComputeInterface &interface) {
          const ktt::DimensionVector blockDimensions(kBlockDimX * kBlockDimY * kBlockDimZ);
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

        auto isWithin = in.data1.data == in.data2.data;

        auto configuration =
          tuner.CreateConfiguration(kernelData.kernelId, {
              { "center", static_cast<uint64_t>(s.GetCenter()) },
              { "inWithin", static_cast<uint64_t>(isWithin) },
              { "TILE", static_cast<uint64_t>(kTile) } });
        tuner.Run(kernelData.kernelId, configuration, {}); // run is blocking call
        // arguments shall be removed once the run is done
        return true;
      };
    };
  }// namespace

  void Correlation_CUDA::Synchronize() { tuner.Synchronize(); }

  // FIXME createApiInitializer is reapeating itself in all gpu implementation, find a way to implement it once
  ktt::ComputeApiInitializer Correlation_CUDA::createApiInitializer(int deviceOrdinal)
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

  ktt::ComputeApiInitializer Correlation_CUDA::createApiInitializer(CUstream stream)
  {
    CudaErrchk(cuInit(0));
    CUcontext context;
    CudaErrchk(cuStreamGetCtx(stream, &context));
    // Create compute API initializer which specifies context and streams that will be utilized by
    // the tuner.
    return ktt::ComputeApiInitializer(context, std::vector<ktt::ComputeQueue>{ stream });
  }


  bool Correlation_CUDA::Init(const OutputData &out,
    const InputData &in,
    const Settings &s)
  {
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

  bool Correlation_CUDA::Execute(const OutputData &out,
    const InputData &in)
  {
    if (!this->IsValid(out, in)) return false;
    return strategy->Execute(out, in, GetSettings(), tuner);
  }

}// namespace correlation
}// namespace umpalumpa
 
