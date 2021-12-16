#include "flexalign_starpu.hpp"
#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_starpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_starpu.hpp>
#include <libumpalumpa/algorithms/correlation/correlation_starpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_starpu.hpp>

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
{}

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