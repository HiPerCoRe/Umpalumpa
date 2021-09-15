#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/utils/system.hpp>
#include <libumpalumpa/utils/cuda.hpp>

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
      inline static const auto kProjectRoot = utils::GetSourceFilePath(
          "../../..");
      static constexpr auto kCompilerOpts = "--std=c++14 -default-device";

      static constexpr size_t kMaxThreads = 512;

      size_t threads;

      KernelData kernelData;

      bool Init(const AExtremaFinder::ResultData &,
        const AExtremaFinder::SearchData &in,
        const Settings &s,
        ktt::Tuner &tuner) override final
      {
        bool canProcess = (s.version == 1) && (s.location == SearchLocation::kEntire)
                          && (s.type == SearchType::kMax) && (s.result == SearchResult::kValue)
                          && (in.data.info.size == in.data.info.paddedSize)
                          && (in.data.dataInfo.type == umpalumpa::data::DataType::kFloat);
        if (canProcess) {
          // how many threads do we need?
          threads = (in.data.info.size.single < kMaxThreads) ? ceilPow2(in.data.info.size.single)
                                                             : kMaxThreads;
          const ktt::DimensionVector blockDimensions(threads);
          const ktt::DimensionVector gridDimensions(in.data.info.size.n);
          kernelData.definitionId = tuner.AddKernelDefinitionFromFile(
            kFindMax1D, kKernelFile, gridDimensions, blockDimensions, {});
          kernelData.kernelId = tuner.CreateSimpleKernel(kFindMax1D, kernelData.definitionId);
          tuner.AddParameter(kernelData.kernelId, "blockSize", std::vector<uint64_t>{ threads });
          tuner.SetCompilerOptions("-I" + kProjectRoot + " " + kCompilerOpts);
        }
        return canProcess;
      }

      std::string GetName() const override final { return kStrategyName; }

      bool Execute(const AExtremaFinder::ResultData &out,
        const AExtremaFinder::SearchData &in,
        const Settings &,
        ktt::Tuner &tuner) override final
      {
        if (!in.data.IsValid() || in.data.IsEmpty() || !out.values.IsValid()
            || out.values.IsEmpty())
          return false;

        // prepare input data
        auto argIn = tuner.AddArgumentVector<float>(in.data.ptr,
          in.data.info.size.total,
          ktt::ArgumentAccessType::ReadOnly,
          ktt::ArgumentMemoryLocation::Unified);

        // prepare output data
        auto argVals = tuner.AddArgumentVector<float>(out.values.ptr,
          out.values.info.size.total,
          ktt::ArgumentAccessType::WriteOnly,
          ktt::ArgumentMemoryLocation::Unified);

        auto argSize = tuner.AddArgumentScalar(in.data.info.size.single);
        // allocate local memory
        auto argLocMem = tuner.AddArgumentLocal<float>(2 * threads * sizeof(float));

        tuner.SetArguments(kernelData.definitionId, { argIn, argVals, argSize, argLocMem });

        // update grid dimension to properly react to batch size
        tuner.SetLauncher(kernelData.kernelId, [this, &in](ktt::ComputeInterface &interface) {
          const ktt::DimensionVector blockDimensions(threads);
          const ktt::DimensionVector gridDimensions(in.data.info.size.n);
          interface.RunKernelAsync(kernelData.definitionId,
            interface.GetAllQueues().at(0),
            gridDimensions,
            blockDimensions);
        });

        auto configuration =
          tuner.CreateConfiguration(kernelData.kernelId, { { "blockSize", threads } });
        tuner.Run(kernelData.kernelId, configuration, {}); // run is blocking call
        // arguments shall be removed once the run is done
        return true;
      };
    };

    struct Strategy2 : public SingleExtremaFinderCUDA::Strategy
    {
      static constexpr auto kFindMaxRect = "findMaxRect";
      static constexpr auto kStrategyName = "Strategy2";
      inline static const auto kKernelFile = utils::GetSourceFilePath(
        "../../../libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda_kernels.cu");
      inline static const auto kProjectRoot = utils::GetSourceFilePath(
          "../../..");
      static constexpr auto kCompilerOpts = "--std=c++14 -default-device";

      //static constexpr size_t kMaxThreads = 512;

      size_t threadsX;
      size_t threadsY;

      KernelData kernelData;

      bool Init(const AExtremaFinder::ResultData &,
        const AExtremaFinder::SearchData &in,
        const Settings &s,
        ktt::Tuner &tuner) override final
      {
        bool canProcess = (s.version == 1) && (s.location == SearchLocation::kRectCenter)
                          && (s.type == SearchType::kMax) && (s.result == SearchResult::kLocation)
                          && (in.data.info.size == in.data.info.paddedSize)
                          && (in.data.dataInfo.type == umpalumpa::data::DataType::kFloat);
        if (canProcess) {
          // how many threads do we need?
          //threads = (in.data.info.size.single < kMaxThreads) ? ceilPow2(in.data.info.size.single)
          //                                                   : kMaxThreads;
          // TODO should be tuned by KTT, for now it is FIXED to work at least somehow
          // block size needs to be power of 2
          threadsX = 64;
          threadsY = 2;
          const ktt::DimensionVector blockDimensions(threadsX, threadsY);
          const ktt::DimensionVector gridDimensions(in.data.info.size.n);
          kernelData.definitionId = tuner.AddKernelDefinitionFromFile(
            kFindMaxRect, kKernelFile, gridDimensions, blockDimensions, {"float"});
          kernelData.kernelId = tuner.CreateSimpleKernel(kFindMaxRect, kernelData.definitionId);
          tuner.AddParameter(kernelData.kernelId, "blockSizeX", std::vector<uint64_t>{ threadsX });
          tuner.AddParameter(kernelData.kernelId, "blockSizeY", std::vector<uint64_t>{ threadsY });
          tuner.AddParameter(kernelData.kernelId, "blockSize", std::vector<uint64_t>{ threadsX * threadsY });
          tuner.SetCompilerOptions("-I" + kProjectRoot + " " + kCompilerOpts);
        }
        return canProcess;
      }

      std::string GetName() const override final { return kStrategyName; }

      bool Execute(const AExtremaFinder::ResultData &out,
        const AExtremaFinder::SearchData &in,
        const Settings &,
        ktt::Tuner &tuner) override final
      {
        if (!in.data.IsValid() || in.data.IsEmpty() || !out.locations.IsValid()
            || out.locations.IsEmpty())
          return false;

        // prepare input data
        auto argIn = tuner.AddArgumentVector<float>(in.data.ptr,
          in.data.info.size.total,
          ktt::ArgumentAccessType::ReadOnly,
          ktt::ArgumentMemoryLocation::Unified);

        auto argInSize = tuner.AddArgumentScalar(in.data.info.size);

        // prepare output data
        auto argVals = tuner.AddArgumentScalar(NULL);
        //auto argVals = tuner.AddArgumentVector<float>(out.values.ptr,
        //  out.values.info.size.total,
        //  ktt::ArgumentAccessType::WriteOnly,
        //  ktt::ArgumentMemoryLocation::Unified);

        // prepare output data
        auto argLocs = tuner.AddArgumentVector<float>(out.locations.ptr,
          out.values.info.size.total,
          ktt::ArgumentAccessType::WriteOnly,
          ktt::ArgumentMemoryLocation::Unified);

        //FIXME these values should be read from settings
        //FIXME offset + rectDim cant be > inSize, add check
        unsigned offsetX = 1;
        unsigned offsetY = 1;
        unsigned rectWidth = 28;
        unsigned rectHeight = 17;

        auto argOffX = tuner.AddArgumentScalar(offsetX);
        auto argOffY = tuner.AddArgumentScalar(offsetY);
        auto argRectWidth = tuner.AddArgumentScalar(rectWidth);
        auto argRectHeight = tuner.AddArgumentScalar(rectHeight);
        // allocate local memory
        auto argLocMem = tuner.AddArgumentLocal<float>(2 * threadsX*threadsY * sizeof(float));

        tuner.SetArguments(kernelData.definitionId, { argIn, argInSize, argVals, argLocs, argOffX, argOffY, argRectWidth, argRectHeight, argLocMem });

        // update grid dimension to properly react to batch size
        tuner.SetLauncher(kernelData.kernelId, [this, &in](ktt::ComputeInterface &interface) {
          const ktt::DimensionVector blockDimensions(threadsX, threadsY);
          const ktt::DimensionVector gridDimensions(in.data.info.size.n);
          interface.RunKernelAsync(kernelData.definitionId,
            interface.GetAllQueues().at(0),
            gridDimensions,
            blockDimensions);
        });

        auto configuration =
          tuner.CreateConfiguration(kernelData.kernelId, {
              { "blockSizeX", threadsX },
              { "blockSizeY", threadsY },
              { "blockSize", threadsX*threadsY } });
        tuner.Run(kernelData.kernelId, configuration, {}); // run is blocking call
        // arguments shall be removed once the run is done
        return true;
      };
    };
  }// namespace

  void SingleExtremaFinderCUDA::Synchronize() { tuner.Synchronize(); }

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

    return tryToAdd(std::make_unique<Strategy1>()) || tryToAdd(std::make_unique<Strategy2>());
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
