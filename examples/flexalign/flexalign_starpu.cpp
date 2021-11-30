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
PhysicalDescriptor FlexAlignStarPU<T>::Create(size_t bytes, DataType type, bool tmp) const
{
  void *ptr = nullptr;
  tmp = false; // FIXME remove
  if (!(tmp || 0 == bytes)) {
    starpu_malloc(&ptr, bytes);
    memset(ptr, 0, bytes);
  }
  auto *handle = new starpu_data_handle_t();
  auto pd = PhysicalDescriptor(ptr, bytes, type, ManagedBy::StarPU, handle);
  StarPUUtils::Register(pd, tmp ? -1 : STARPU_MAIN_RAM);
  return pd;
}

template<typename T> void FlexAlignStarPU<T>::Remove(const PhysicalDescriptor &pd) const
{
  StarPUUtils::Unregister(pd, StarPUUtils::UnregisterType::kBlockingCopyToHomeNode);
  delete StarPUUtils::GetHandle(pd);
  starpu_free(pd.GetPtr());
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