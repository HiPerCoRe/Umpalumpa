#include <functional>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_gpu.hpp>
#include <libumpalumpa/utils/ktt.hpp>
#include <libumpalumpa/utils/logger.hpp>
#include <libumpalumpa/utils/system.hpp>

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

    // template<typename T, typename C>
    // static void sFindUniversal(const C &comp,
    //   T startVal,
    //   // const GPU &gpu,
    //   const umpalumpa::data::Size &size,
    //   const T *__restrict__ d_data,
    //   float *__restrict__ d_positions,
    //   T *__restrict__ d_values)
    // {
    //   // check input
    //   assert(size.total <= std::numeric_limits<unsigned>::max());// indexing overflow in the
    //   kernel

    //   // create threads / blocks
    //   size_t maxThreads = 512;
    //   size_t threads = (size.single < maxThreads) ? ceilPow2(size.single) : maxThreads;
    //   dim3 dimBlock(threads, 1, 1);
    //   dim3 dimGrid(dims.n(), 1, 1);
    //   // auto stream = *(cudaStream_t *)gpu.stream(); // FIXME add stream

    //   // for each thread, we need two variables in shared memory
    //   size_t smemSize = 2 * threads * sizeof(T);
    //   switch (threads) {
    //   case 512:
    //     return findUniversal<T, 512><<<dimGrid, dimBlock, smemSize>>>(
    //       comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
    //   case 256:
    //     return findUniversal<T, 256><<<dimGrid, dimBlock, smemSize>>>(
    //       comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
    //   case 128:
    //     return findUniversal<T, 128><<<dimGrid, dimBlock, smemSize>>>(
    //       comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
    //   case 64:
    //     return findUniversal<T, 64><<<dimGrid, dimBlock, smemSize>>>(
    //       comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
    //   case 32:
    //     return findUniversal<T, 32><<<dimGrid, dimBlock, smemSize>>>(
    //       comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
    //   case 16:
    //     return findUniversal<T, 16><<<dimGrid, dimBlock, smemSize>>>(
    //       comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
    //   case 8:
    //     return findUniversal<T, 8><<<dimGrid, dimBlock, smemSize>>>(
    //       comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
    //   case 4:
    //     return findUniversal<T, 4><<<dimGrid, dimBlock, smemSize>>>(
    //       comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
    //   case 2:
    //     return findUniversal<T, 2><<<dimGrid, dimBlock, smemSize>>>(
    //       comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
    //   case 1:
    //     return findUniversal<T, 1><<<dimGrid, dimBlock, smemSize>>>(
    //       comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
    //   default:
    //     REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Unsupported number of threads");
    //   }
    // }


    struct Strategy1
    {
      static bool CanRun(__attribute__((unused)) const ResultData &out,
        const SearchData &in,
        const Settings &settings)
      {
        return (settings.version == 1) && (in.info.size == in.info.paddedSize)
               && (settings.location == SearchLocation::kEntire)
               && (settings.type == SearchType::kMax) && (settings.result == SearchResult::kValue)
               && (in.dataInfo.type == umpalumpa::data::DataType::kFloat);
      }

      static bool Run(const ResultData &out, const SearchData &in, const Settings &settings)
      {
        if (settings.dryRun) return true;
        if (nullptr == in.data || nullptr == out.values->data) return false;


        ktt::DeviceIndex deviceIndex = 0;
        // std::string relPath =
        //   utils::GetExecPath()
        //   +
        //   "../../../libumpalumpa/algorithms/extrema_finder/single_extrema_finder_gpu_kernels.cu";
        // spdlog::error("{}\n", relPath);
        // std::string kernelFile = utils::Canonize(relPath);
        // spdlog::error("{}", kernelFile);
        std::string kernelFile =
          "../libumpalumpa/algorithms/extrema_finder/single_extrema_finder_gpu_kernels.cu";

        // create threads / blocks
        constexpr size_t maxThreads = 512;
        size_t threads =
          (in.info.size.single < maxThreads) ? ceilPow2(in.info.size.single) : maxThreads;
        const ktt::DimensionVector blockDimensions(threads);
        const ktt::DimensionVector gridDimensions(in.info.size.n);
        ktt::Tuner tuner(0, deviceIndex, ktt::ComputeApi::CUDA);

        const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile(
          "findMax1D", kernelFile, gridDimensions, blockDimensions);

        // Add new kernel arguments to tuner. Argument data is copied from std::vector containers.
        // Specify whether the arguments are used as input or output. KTT returns handle to the
        // newly added argument, which can be used to reference it in other API methods.

        std::vector<float> tmpIn(in.info.size.total);
        memcpy(tmpIn.data(), in.data, in.dataInfo.bytes);
        auto argIn = tuner.AddArgumentVector(tmpIn, ktt::ArgumentAccessType::ReadOnly);
        std::vector<float> tmpOut(out.values->info.size.total);
        memcpy(tmpOut.data(), out.values->data, out.values->dataInfo.bytes);
        auto argVals = tuner.AddArgumentVector(tmpOut, ktt::ArgumentAccessType::WriteOnly);
        auto argSize =
          tuner.AddArgumentScalar(in.info.size.single);// FIXME vykopirovat z tmp zpatky vysledky

        auto argLocMem = tuner.AddArgumentLocal<float>(2 * threads * sizeof(float));


        // Set arguments for the kernel definition. The order of argument ids must match the order
        // of arguments inside corresponding CUDA kernel function.
        tuner.SetArguments(definition, { argIn, argVals, argSize, argLocMem });

        // Create simple kernel from the specified definition. Specify name which will be used
        // during logging and output operations. In more complex scenarios, kernels can have
        // multiple definitions. Definitions can be shared between multiple kernels.
        const ktt::KernelId kernel = tuner.CreateSimpleKernel("Find max kernel", definition);

        tuner.AddParameter(kernel, "blockSize", std::vector<uint64_t>{ threads });
        // Set time unit used during printing of kernel duration. The default time unit is
        // milliseconds, but since computation in this tutorial is very short, microseconds are used
        // instead.
        tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

        // Run the specified kernel. The second argument is related to kernel tuning and will be
        // described in further tutorials. In this case, it remains empty. The third argument is
        // used to retrieve the kernel output. For each kernel argument that is retrieved, one
        // BufferOutputDescriptor must be specified. Each of these descriptors contains id of the
        // retrieved argument and memory location where the argument data will be stored.
        // Optionally, it can also include number of bytes to be retrieved, if only a part of the
        // argument is needed. Here, the data is stored back into result buffer which was created
        // earlier. Note that the memory location size needs to be equal or greater than the
        // retrieved argument size.

        auto configuration = tuner.CreateConfiguration(kernel, { { "blockSize", threads } });

        tuner.RunKernel(
          kernel, configuration, { ktt::BufferOutputDescriptor(argVals, out.values->data) });

        // sFindUniversal([] __device__(float l, float r) { return l > r; },
        //   std::numeric_limits<float>::lowest(),// FIXME remove
        //   // gpu,
        //   in.info.size,
        //   reinterpret_cast<float *>(in.data),
        //   nullptr,// d_positions,
        //   reinterpret_cast<float *>(out.values->data));

        return true;
      }
    };
  }// namespace

  bool SingleExtremaFinderGPU::Execute(const ResultData &out,
    const SearchData &in,
    const Settings &settings)
  {
    if (!this->IsValid(out, in, settings)) return false;
    if (Strategy1::CanRun(out, in, settings)) return Strategy1::Run(out, in, settings);
    return false;// no strategy could process these data
  }
}// namespace extrema_finder
}// namespace umpalumpa