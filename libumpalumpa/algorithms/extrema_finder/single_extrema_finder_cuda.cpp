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

    inline static const auto kKernelFile = utils::GetSourceFilePath(
      "libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda_kernels.cu");

    struct Strategy1 : public SingleExtremaFinderCUDA::Strategy
    {
      static constexpr auto kFindMax1D = "findMax1D";
      static constexpr size_t kernelDataIndex = 0;
      static constexpr size_t kMaxThreads = 512;

      size_t threads;

      bool Init(const AExtremaFinder::ResultData &,
        const AExtremaFinder::SearchData &in,
        const Settings &s,
        utils::KTTHelper &helper) override final
      {
        bool canProcess = (s.version == 1) && (s.location == SearchLocation::kEntire)
                          && (s.type == SearchType::kMax) && (s.result == SearchResult::kValue)
                          && (!in.data.info.IsPadded())
                          && (in.data.dataInfo.type == umpalumpa::data::DataType::kFloat);
        if (canProcess) {
          // how many threads do we need?
          threads = (in.data.info.GetSize().single < kMaxThreads) ? ceilPow2(in.data.info.GetSize().single)
                                                             : kMaxThreads;
          const ktt::DimensionVector blockDimensions(threads);
          const ktt::DimensionVector gridDimensions(in.data.info.GetSize().n);
          auto &tuner = helper.GetTuner();
          // ensure that we have the kernel loaded to KTT
          // this has to be done in critical section, as multiple instances of this algorithm
          // might run on the same worker
          std::lock_guard<std::mutex> lck(helper.GetMutex());
          auto &kernelData = helper.GetKernelData(GetFullName());
          auto it = kernelData.find(kernelDataIndex);
          if (kernelData.end() == it) {
            auto definitionId = tuner.AddKernelDefinitionFromFile(
              kFindMax1D, kKernelFile, gridDimensions, blockDimensions, {});
            auto kernelId = tuner.CreateSimpleKernel(kFindMax1D, definitionId);
            tuner.AddParameter(kernelId, "blockSize", std::vector<uint64_t>{ threads });
            // register kernel data
            kernelData[kernelDataIndex] = { { definitionId }, { kernelId } };
          }
        }
        return canProcess;
      }

      std::string GetName() const override final { return "Strategy1"; }

      bool Execute(const AExtremaFinder::ResultData &out,
        const AExtremaFinder::SearchData &in,
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
        // allocate local memory
        auto argLocMem = tuner.AddArgumentLocal<float>(2 * threads * sizeof(float));

        auto definitionId =
          helper.GetKernelData(GetFullName()).at(kernelDataIndex).definitionIds[0];
        tuner.SetArguments(definitionId, { argIn, argVals, argSize, argLocMem });

        // update grid dimension to properly react to batch size
        auto kernelId = helper.GetKernelData(GetFullName()).at(kernelDataIndex).kernelIds[0];
        tuner.SetLauncher(kernelId, [this, &in, definitionId](ktt::ComputeInterface &interface) {
          const ktt::DimensionVector blockDimensions(threads);
          const ktt::DimensionVector gridDimensions(in.data.info.GetSize().n);
          interface.RunKernelAsync(
            definitionId, interface.GetAllQueues().at(0), gridDimensions, blockDimensions);
        });

        auto configuration = tuner.CreateConfiguration(kernelId, { { "blockSize", threads } });
        tuner.Run(kernelId, configuration, {});// run is blocking call
        // arguments shall be removed once the run is done
        return true;
      };
    };

    struct Strategy2 : public SingleExtremaFinderCUDA::Strategy
    {
      static constexpr auto kFindMaxRect = "findMaxRect";
      static constexpr size_t kernelDataIndex = 0;

      size_t threadsX;
      size_t threadsY;

      bool Init(const AExtremaFinder::ResultData &,
        const AExtremaFinder::SearchData &in,
        const Settings &s,
        utils::KTTHelper &helper) override final
      {
        bool canProcess = (s.version == 1) && (s.location == SearchLocation::kRectCenter)
                          && (s.type == SearchType::kMax) && (s.result == SearchResult::kLocation)
                          && (!in.data.info.IsPadded())
                          && (in.data.dataInfo.type == umpalumpa::data::DataType::kFloat);
        if (canProcess) {
          // TODO should be tuned by KTT, for now it is FIXED to work at least somehow
          // block size needs to be power of 2
          threadsX = 64;
          threadsY = 2;
          const ktt::DimensionVector blockDimensions(threadsX, threadsY);
          const ktt::DimensionVector gridDimensions(in.data.info.GetSize().n);
          auto &tuner = helper.GetTuner();
          // ensure that we have the kernel loaded to KTT
          // this has to be done in critical section, as multiple instances of this algorithm
          // might run on the same worker
          std::lock_guard<std::mutex> lck(helper.GetMutex());
          auto &kernelData = helper.GetKernelData(GetFullName());
          auto it = kernelData.find(kernelDataIndex);
          if (kernelData.end() == it) {
            auto definitionId = tuner.AddKernelDefinitionFromFile(
              kFindMaxRect, kKernelFile, gridDimensions, blockDimensions, { "float" });
            auto kernelId = tuner.CreateSimpleKernel(kFindMaxRect, definitionId);
            tuner.AddParameter(kernelId, "blockSizeX", std::vector<uint64_t>{ threadsX });
            tuner.AddParameter(kernelId, "blockSizeY", std::vector<uint64_t>{ threadsY });
            tuner.AddParameter(kernelId, "blockSize", std::vector<uint64_t>{ threadsX * threadsY });
            // register kernel data
            kernelData[kernelDataIndex] = { { definitionId }, { kernelId } };
          }
        }
        return canProcess;
      }

      std::string GetName() const override final { return "Strategy2"; }

      bool Execute(const AExtremaFinder::ResultData &out,
        const AExtremaFinder::SearchData &in,
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

        //FIXME these values should be read from settings
        //FIXME offset + rectDim cant be > inSize, add check
        // Compute the area to search in
        size_t searchRectWidth = 28;
        size_t searchRectHeight = 17;
        size_t searchRectOffsetX =
          (in.data.info.GetPaddedSize().x - searchRectWidth) / 2;
        size_t searchRectOffsetY =
          (in.data.info.GetPaddedSize().y - searchRectHeight) / 2;

        auto argOffX = tuner.AddArgumentScalar(searchRectOffsetX);
        auto argOffY = tuner.AddArgumentScalar(searchRectOffsetY);
        auto argRectWidth = tuner.AddArgumentScalar(searchRectWidth);
        auto argRectHeight = tuner.AddArgumentScalar(searchRectHeight);
        // allocate local memory
        auto argLocMem = tuner.AddArgumentLocal<float>(2 * threadsX * threadsY * sizeof(float));

        auto definitionId =
          helper.GetKernelData(GetFullName()).at(kernelDataIndex).definitionIds[0];
        tuner.SetArguments(definitionId,
          { argIn,
            argInSize,
            argVals,
            argLocs,
            argOffX,
            argOffY,
            argRectWidth,
            argRectHeight,
            argLocMem });

        auto kernelId = helper.GetKernelData(GetFullName()).at(kernelDataIndex).kernelIds[0];
        // update grid dimension to properly react to batch size
        tuner.SetLauncher(kernelId, [this, &in, definitionId](ktt::ComputeInterface &interface) {
          const ktt::DimensionVector blockDimensions(threadsX, threadsY);
          const ktt::DimensionVector gridDimensions(in.data.info.GetSize().n);
          interface.RunKernelAsync(
            definitionId, interface.GetAllQueues().at(0), gridDimensions, blockDimensions);
        });

        auto configuration = tuner.CreateConfiguration(kernelId,
          { { "blockSizeX", threadsX },
            { "blockSizeY", threadsY },
            { "blockSize", threadsX * threadsY } });
        tuner.Run(kernelId, configuration, {});// run is blocking call
        // arguments shall be removed once the run is done
        return true;
      };
    };
  }// namespace

  void SingleExtremaFinderCUDA::Synchronize() { GetHelper().GetTuner().Synchronize(); }

  bool SingleExtremaFinderCUDA::Init(const ResultData &out,
    const SearchData &in,
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

  bool SingleExtremaFinderCUDA::Execute(const ResultData &out,
    const SearchData &in,
    const Settings &settings)
  {
    if (!this->IsValid(out, in, settings)) return false;
    return strategy->Execute(out, in, settings, GetHelper());
  }

}// namespace extrema_finder
}// namespace umpalumpa
