#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/utils/system.hpp>
#include <cuda.h>

namespace umpalumpa {
namespace extrema_finder {

  namespace {// to avoid poluting

    size_t ceilPow2(size_t x)
    {
      if (x <= 1) return 1;
      size_t power = 2;
      x--;
      while (x >>= 1) power <<= 1;
      return power;
    }

    struct Strategy1 : public SingleExtremaFinderCUDA::Strategy
    {
      static constexpr auto kFindMax1D = "findMax1D";
      static constexpr auto kStrategyName = "Strategy1";
      inline static const auto kKernelFile = utils::GetSourceFilePath(
        "../../../libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda_kernels.cu");

      static constexpr size_t kMaxThreads = 512;

      size_t threads;

      KernelData kernelData;

      bool Init(const ResultData &,
        const SearchData &in,
        const Settings &s,
        ktt::Tuner &tuner) override final
      {
        bool canProcess = (s.version == 1) && (s.location == SearchLocation::kEntire)
                          && (s.type == SearchType::kMax) && (s.result == SearchResult::kValue)
                          && (in.info.size == in.info.paddedSize)
                          && (in.dataInfo.type == umpalumpa::data::DataType::kFloat);
        if (canProcess) {
          // how many threads do we need?
          threads =
            (in.info.size.single < kMaxThreads) ? ceilPow2(in.info.size.single) : kMaxThreads;
          const ktt::DimensionVector blockDimensions(threads);
          const ktt::DimensionVector gridDimensions(in.info.size.n);
          kernelData.definitionId = tuner.AddKernelDefinitionFromFile(
            kFindMax1D, kKernelFile, gridDimensions, blockDimensions);
          kernelData.kernelId = tuner.CreateSimpleKernel(kFindMax1D, kernelData.definitionId);
          tuner.AddParameter(kernelData.kernelId, "blockSize", std::vector<uint64_t>{ threads });
        }
        return canProcess;
      }

      std::string GetName() const override final { return kStrategyName; }

      bool Execute(const ResultData &out,
        const SearchData &in,
        const Settings &settings,
        ktt::Tuner &tuner) override final
      {
        if (settings.dryRun) return true;
        if ((nullptr == in.data) || (nullptr == out.values->data)) return false;

        // prepare input data
        auto argIn = tuner.AddArgumentVector<float>(in.data,
          in.info.size.total,
          ktt::ArgumentAccessType::ReadOnly,
          ktt::ArgumentMemoryLocation::Unified);

        // prepare output data
        auto argVals = tuner.AddArgumentVector<float>(out.values->data,
          out.values->info.size.total,
          ktt::ArgumentAccessType::WriteOnly,
          ktt::ArgumentMemoryLocation::Unified);

        auto argSize = tuner.AddArgumentScalar(in.info.size.single);
        // allocate local memory
        auto argLocMem = tuner.AddArgumentLocal<float>(2 * threads * sizeof(float));

        tuner.SetArguments(kernelData.definitionId, { argIn, argVals, argSize, argLocMem });

        // update grid dimension to properly react to batch size
        tuner.SetLauncher(kernelData.kernelId, [this, &in](ktt::ComputeInterface &interface) {
          const ktt::DimensionVector blockDimensions(threads);
          const ktt::DimensionVector gridDimensions(in.info.size.n);
          interface.RunKernel(kernelData.definitionId, gridDimensions, blockDimensions);
        });

        auto configuration =
          tuner.CreateConfiguration(kernelData.kernelId, { { "blockSize", threads } });
        tuner.RunKernel(kernelData.kernelId,
          configuration,
          { ktt::BufferOutputDescriptor(argVals, out.values->data) });
        return true;
      };
    };
  }// namespace


  ktt::ComputeApiInitializer SingleExtremaFinderCUDA::createApiInitializer(int deviceOrdinal)
  {
    cuInit(0);// FIXME add error handling

    CUdevice device;
    cuDeviceGet(&device, deviceOrdinal);

    CUcontext context;
    cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);

    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);

    // Create compute API initializer which specifies context and streams that will be utilized by
    // the tuner.
    return ktt::ComputeApiInitializer(context, std::vector<ktt::ComputeQueue>{ stream });
  }

  bool SingleExtremaFinderCUDA::Init(const ResultData &out,
    const SearchData &in,
    const Settings &settings)
  {
    auto tryToAdd = [this, &out, &in, &settings](auto i) {
      bool canAdd = i->Init(out, in, settings, tuner);
      if (canAdd) {
        spdlog::debug("Found valid strategy {}", i->GetName());
        strategy = std::move(i);
      }
      return canAdd;
    };

    return tryToAdd(std::make_unique<Strategy1>()) || false;
  }

  bool SingleExtremaFinderCUDA::Execute(const ResultData &out,
    const SearchData &in,
    const Settings &settings)
  {
    if (!this->IsValid(out, in, settings)) return false;
    return strategy->Execute(out, in, settings, tuner);
  }

}// namespace extrema_finder
}// namespace umpalumpa