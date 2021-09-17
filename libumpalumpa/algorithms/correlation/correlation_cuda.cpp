#include <libumpalumpa/algorithms/correlation/correlation_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/utils/system.hpp>
#include <libumpalumpa/utils/cuda.hpp>

namespace umpalumpa {
namespace correlation {

  namespace {// to avoid poluting
    static constexpr auto kCompileFlags = "--std=c++14 -default-device";
    inline static const auto kKernelFile =
      utils::GetSourceFilePath("libumpalumpa/algorithms/correlation/correlation_cuda_kernels.cu");

    struct Strategy1 : public Correlation_CUDA::Strategy
    {
      // FIXME improve name of the kernel and this variable
      static constexpr auto kTMP = "correlate2D";
      static constexpr size_t kernelDataIndex = 0;
      // Currently we create one thread per each pixel of a single image. Each thread processes
      // same pixel of all images. The other option for 2D images is to map N dimension to the
      // Z dimension, ie. create more threads, each thread processing fewer images.
      // FIXME  this should be tuned by the KTT
      static constexpr auto kBlockDimX = 32;
      static constexpr auto kBlockDimY = 32;
      static constexpr auto kBlockDimZ = 1;

      static constexpr auto kTile = 8;

      inline size_t ComputeDimension(size_t l, int r) const
      {
        return static_cast<size_t>(std::ceil(static_cast<float>(l) / static_cast<float>(r)));
      }

      bool Init(const Correlation_CUDA::OutputData &out,
        const Correlation_CUDA::InputData &in,
        const Settings &s,
        utils::KTTHelper &helper) override final
      {
        bool canProcess = true;// FIXME

        if (canProcess) {
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
          auto &kernelData = helper.GetKernelData(GetFullName());
          auto it = kernelData.find(kernelDataIndex);
          if (kernelData.end() == it) {
            auto definitionId = tuner.AddKernelDefinitionFromFile(
              kTMP, kKernelFile, gridDimensions, blockDimensions, { "float2" });
            auto kernelId = tuner.CreateSimpleKernel(kTMP, definitionId);
            tuner.AddParameter(kernelId, "center", std::vector<uint64_t>{ s.GetCenter() });
            tuner.AddParameter(
              kernelId, "isWithin", std::vector<uint64_t>{ in.data1.ptr == in.data2.ptr });
            tuner.AddParameter(kernelId, "TILE", std::vector<uint64_t>{ kTile });
            // register kernel data
            kernelData[kernelDataIndex] = { { definitionId }, { kernelId } };
          }
        }
        return canProcess;
      }

      std::string GetName() const override final { return "Strategy1"; }

      bool Execute(const Correlation_CUDA::OutputData &out,
        const Correlation_CUDA::InputData &in,
        const Settings &s,
        utils::KTTHelper &helper) override final
      {
        if (!in.data1.IsValid() || in.data1.IsEmpty() || !out.data.IsValid()// FIXME refactor
            || out.data.IsEmpty())
          return false;

        auto &tuner = helper.GetTuner();
        // prepare input data1
        auto argIn1 = tuner.AddArgumentVector<float2>(in.data1.ptr,
          in.data1.info.GetSize().total,
          ktt::ArgumentAccessType::ReadOnly,// FIXME these information should be stored in the
                                            // physical descriptor
          ktt::ArgumentMemoryLocation::Unified);// ^

        auto argIn2 = tuner.AddArgumentVector<float2>(in.data2.ptr,
          in.data2.info.GetSize().total,
          ktt::ArgumentAccessType::ReadOnly,// FIXME these information should be stored in the
                                            // physical descriptor
          ktt::ArgumentMemoryLocation::Unified);// ^

        // prepare output data1
        auto argOut = tuner.AddArgumentVector<float2>(out.data.ptr,
          out.data.info.GetSize().total,
          ktt::ArgumentAccessType::WriteOnly,// FIXME these information should be stored in the
                                             // physical descriptor
          ktt::ArgumentMemoryLocation::Unified);// ^

        auto inSize = tuner.AddArgumentScalar(in.data1.info.GetSize());
        auto in2N = tuner.AddArgumentScalar(static_cast<int>(in.data2.info.GetSize().n));

        auto definitionId =
          helper.GetKernelData(GetFullName()).at(kernelDataIndex).definitionIds[0];
        tuner.SetArguments(definitionId, { argOut, argIn1, inSize, argIn2, in2N });

        // update grid dimension to properly react to batch size
        auto kernelId = helper.GetKernelData(GetFullName()).at(kernelDataIndex).kernelIds[0];
        tuner.SetLauncher(kernelId, [this, &out, definitionId](ktt::ComputeInterface &interface) {
          const ktt::DimensionVector blockDimensions(kBlockDimX, kBlockDimY, kBlockDimZ);
          const auto &size = out.data.info.GetPaddedSize();
          const ktt::DimensionVector gridDimensions(ComputeDimension(size.x, kBlockDimX),
            ComputeDimension(size.y, kBlockDimY),
            ComputeDimension(size.z, kBlockDimZ));
          interface.RunKernelAsync(
            definitionId, interface.GetAllQueues().at(0), gridDimensions, blockDimensions);
        });

        auto isWithin = in.data1.ptr == in.data2.ptr;

        auto configuration = tuner.CreateConfiguration(kernelId,
          { { "center", static_cast<uint64_t>(s.GetCenter()) },
            { "isWithin", static_cast<uint64_t>(isWithin) },
            { "TILE", static_cast<uint64_t>(kTile) } });
        tuner.Run(kernelId, configuration, {});// run is blocking call
        // arguments shall be removed once the run is done
        return true;
      };
    };
  }// namespace

  void Correlation_CUDA::Synchronize() { GetHelper().GetTuner().Synchronize(); }

  bool Correlation_CUDA::Init(const OutputData &out, const InputData &in, const Settings &s)
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

  bool Correlation_CUDA::Execute(const OutputData &out, const InputData &in)
  {
    if (!this->IsValid(out, in)) return false;
    return strategy->Execute(out, in, GetSettings(), GetHelper());
  }

}// namespace correlation
}// namespace umpalumpa
