#include "flexalign_starpu.hpp"
#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/operations/fourier_transformation/fft_starpu.hpp>
#include <libumpalumpa/operations/fourier_processing/fp_starpu.hpp>
#include <libumpalumpa/operations/correlation/correlation_starpu.hpp>
#include <libumpalumpa/operations/extrema_finder/single_extrema_finder_starpu.hpp>
#include <libumpalumpa/utils/cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

#include <sys/sysinfo.h>

using umpalumpa::utils::StarPUUtils;
using umpalumpa::data::ManagedBy;
using namespace umpalumpa;

template<typename T>
FlexAlignStarPU<T>::FlexAlignStarPU()
  : forwardFFTOp(std::make_unique<fourier_transformation::FFTStarPU>()),
    inverseFFTOp(std::make_unique<fourier_transformation::FFTStarPU>()),
    cropOp(std::make_unique<fourier_processing::FPStarPU>()),
    corrOp(std::make_unique<correlation::Correlation_StarPU>()),
    extremaFinderOp(std::make_unique<extrema_finder::SingleExtremaFinderStarPU>())
{
  SetAvailableBytesRAM();
  SetAvailableBytesCUDA();
  STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "StarPU init");
  thr = std::make_unique<std::thread>([this]() { RemoveFromQueue(); });
}

template<typename T> void FlexAlignStarPU<T>::SetAvailableBytesRAM()
{
  struct sysinfo info;
  sysinfo(&info);
  // FIXME this can be incorrect due to memory used as cache
  // see e.g. https://scoutapm.com/blog/determining-free-memory-on-linux
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
    putenv(strncpy(MaxBytesCUDA[i], src.c_str(), sizeof(MaxBytesCUDA[i])));
  }
}

template<typename T> FlexAlignStarPU<T>::~FlexAlignStarPU()
{
  // operations must be deleted before StarPU is turned off
  forwardFFTOp.release();
  inverseFFTOp.release();
  cropOp.release();
  corrOp.release();
  extremaFinderOp.release();
  {
    keepWorking = false;// tell thread that we want to finish
    workAvailable.notify_one();// wake it up
    thr->join();// wait till it's done
  }
  if (!toRemove.empty()) { spdlog::error("Some Physical Descriptors were not removed!"); }
  starpu_shutdown();
}

template<typename T>
PhysicalDescriptor
  FlexAlignStarPU<T>::CreatePD(size_t bytes, DataType type, bool copyInRAM, bool pinned)
{
  void *ptr = nullptr;
  if (copyInRAM) {
    starpu_memory_allocate(STARPU_MAIN_RAM, bytes, STARPU_MEMORY_WAIT);
    starpu_memory_wait_available(STARPU_MAIN_RAM, bytes);
    auto flags = STARPU_MALLOC_COUNT | (pinned ? STARPU_MALLOC_PINNED : 0);
    starpu_malloc_flags(&ptr, bytes, flags);
    memset(ptr, 0, bytes);
  }
  auto *handle = new starpu_data_handle_t();
  auto pd = PhysicalDescriptor(ptr, bytes, type, ManagedBy::StarPU, handle);
  pd.SetPinned(pinned);
  StarPUUtils::Register(pd, copyInRAM ? STARPU_MAIN_RAM : -1);
  return pd;
}

template<typename T> void FlexAlignStarPU<T>::RemoveFromQueue()
{
  while (true) {
    std::unique_lock lock(mutex);
    while (toRemove.empty()) {
      if (!keepWorking) return;
      workAvailable.wait(lock);
    }
    // make a local copy
    auto data = toRemove.front();
    toRemove.pop();
    // and unlock the queue, so it's not blocked till we're done here
    lock.unlock();
    auto *handle = reinterpret_cast<starpu_data_handle_t *>(data.handle);
    starpu_data_unregister_no_coherency(*handle);
    if (nullptr != data.ptr) {
      auto flags = STARPU_MALLOC_COUNT | (data.isPinned ? STARPU_MALLOC_PINNED : 0);
      starpu_free_flags(data.ptr, data.bytes, flags);
    }
    delete handle;
  }
}

template<typename T> void FlexAlignStarPU<T>::RemovePD(const PhysicalDescriptor &pd)
{
  if (nullptr == pd.GetPtr()) {
    // these are managed internally by StarPU, so just mark them as removable
    StarPUUtils::Unregister(pd, StarPUUtils::UnregisterType::kSubmitNoCopy);
    delete StarPUUtils::GetHandle(pd);
  } else {
    // we need to make sure that the data is not in use before we delete it
    std::unique_lock lock(mutex);
    auto wasEmpty = toRemove.empty();
    toRemove.push(pd);
    lock.unlock();
    workAvailable.notify_one();
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