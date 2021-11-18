#include "flexalign_cpu.hpp"
#include <libumpalumpa/algorithms/fourier_transformation/fft_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu.hpp>
#include <libumpalumpa/algorithms/correlation/correlation_cpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>

using umpalumpa::data::ManagedBy;
using namespace umpalumpa;

template<typename T>
FlexAlignCPU<T>::FlexAlignCPU()
  : forwardFFTAlg(std::make_unique<fourier_transformation::FFTCPU>()),
    inverseFFTAlg(std::make_unique<fourier_transformation::FFTCPU>()),
    cropAlg(std::make_unique<fourier_processing::FPCPU>()),
    corrAlg(std::make_unique<correlation::Correlation_CPU>()),
    extremaFinderAlg(std::make_unique<extrema_finder::SingleExtremaFinderCPU>())
{}

template<typename T>
PhysicalDescriptor FlexAlignCPU<T>::Create(size_t bytes, DataType type, bool tmp) const
{
  void *ptr = nullptr;
  tmp = false; // FIXME remove
  if (!(tmp || 0 == bytes)) {
    ptr = calloc(bytes, 1);
  }
  return PhysicalDescriptor(ptr, bytes, type, ManagedBy::Manually, nullptr);
}

template<typename T> void FlexAlignCPU<T>::Remove(const PhysicalDescriptor &pd) const
{
  free(pd.GetPtr());
}

template class FlexAlignCPU<float>;