#include "flexalign_starpu.hpp"
#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_starpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_starpu.hpp>
#include <libumpalumpa/algorithms/correlation/correlation_starpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_starpu.hpp>
#include <libumpalumpa/utils/cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

#include <sys/sysinfo.h>

using umpalumpa::utils::StarPUUtils;
using umpalumpa::data::ManagedBy;
using namespace umpalumpa;

template<typename T>
FlexAlignStarPU<T>::FlexAlignStarPU()
  : forwardFFTAlg(std::make_unique<fourier_transformation::FFTStarPU>()),
    inverseFFTAlg(std::make_unique<fourier_transformation::FFTStarPU>()),
    cropAlg(std::make_unique<fourier_processing::FPStarPU>()),
    corrAlg(std::make_unique<correlation::Correlation_StarPU>()),
    extremaFinderAlg(std::make_unique<extrema_finder::SingleExtremaFinderStarPU>())
{
  SetAvailableBytesRAM();
  SetAvailableBytesCUDA();
  STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "StarPU init");
}

template<typename T> void FlexAlignStarPU<T>::SetAvailableBytesRAM()
{
  struct sysinfo info;
  sysinfo(&info);
  auto bytes = info.freeram * info.mem_unit;
  auto MB = bytes / 1048576;
  spdlog::info("Available {}MB of RAM", MB);
  MB = static_cast<decltype(MB)>(MB * 0.95);
  spdlog::warn("Limiting max usage to {}MB", MB);
  // must be static so that it can be referenced later during runtime
  static auto var = "STARPU_LIMIT_CPU_MEM=" + std::to_string(MB);
  putenv(var.data());
}

template<typename T> void FlexAlignStarPU<T>::SetAvailableBytesCUDA()
{

  // must be static so that it can be referenced later during runtime
  static char MaxBytesCUDA[256][256];
  int noOfDevices = 0;
  CudaErrchk(cudaGetDeviceCount(&noOfDevices));
  assert(noOfDevices <= sizeof(MaxBytesCUDA));
  for (int i = 0; i < noOfDevices; ++i) {
    CudaErrchk(cudaSetDevice(i));
    int id;
    CudaErrchk(cudaGetDevice(&id));
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    CudaErrchk(cudaMemGetInfo(&freeBytes, &totalBytes));
    auto MB = freeBytes / 1048576;
    spdlog::info("Available {}MB of memory on CUDA device {}", MB, id);
    MB = static_cast<decltype(MB)>(MB * 0.95);
    spdlog::warn("Limiting max usage to {}MB", MB);
    auto src = "STARPU_LIMIT_CUDA_" + std::to_string(id) + "_MEM=" + std::to_string(MB);
    auto *dst = MaxBytesCUDA[i];
    strncpy(dst, src.c_str(), sizeof(dst));
    putenv(dst);
  }
}

template<typename T> FlexAlignStarPU<T>::~FlexAlignStarPU()
{
  // algorithms must be deleted before StarPU is turned off
  forwardFFTAlg.release();
  inverseFFTAlg.release();
  cropAlg.release();
  corrAlg.release();
  extremaFinderAlg.release();
  starpu_shutdown();
}

template<typename T>
PhysicalDescriptor FlexAlignStarPU<T>::CreatePD(size_t bytes, DataType type, bool copyInRAM)
{
  void *ptr = nullptr;
  if (copyInRAM) {
    starpu_memory_allocate(STARPU_MAIN_RAM, bytes, STARPU_MEMORY_WAIT);
    starpu_memory_wait_available(STARPU_MAIN_RAM, bytes);
    starpu_malloc_flags(&ptr, bytes, STARPU_MALLOC_COUNT);
    memset(ptr, 0, bytes);
  }
  auto *handle = new starpu_data_handle_t();
  auto pd = PhysicalDescriptor(ptr, bytes, type, ManagedBy::StarPU, handle);
  StarPUUtils::Register(pd, copyInRAM ? STARPU_MAIN_RAM : -1);
  return pd;
}

template<typename T> void FlexAlignStarPU<T>::RemovePD(const PhysicalDescriptor &pd) const
{
  StarPUUtils::Unregister(pd, StarPUUtils::UnregisterType::kSubmitNoCopy);
  // don't release the handle, some task might still use it
  // we can release the pointer, because either the data should be already processed,
  // or not allocated at this node at all
  delete StarPUUtils::GetHandle(pd);
  if (nullptr != pd.GetPtr()) {
    starpu_free_flags(pd.GetPtr(), pd.GetBytes(), STARPU_MALLOC_COUNT);
  }
}

template<typename T> void FlexAlignStarPU<T>::Acquire(const PhysicalDescriptor &pd) const
{
  starpu_data_acquire(*StarPUUtils::GetHandle(pd), STARPU_RW);
}

template<typename T> void FlexAlignStarPU<T>::Release(const PhysicalDescriptor &pd) const
{
  starpu_data_release(*StarPUUtils::GetHandle(pd));
}

template class FlexAlignStarPU<float>;