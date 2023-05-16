#include "flexalign_cpu.hpp"
#include <libumpalumpa/operations/fourier_transformation/fft_cpu.hpp>
#include <libumpalumpa/operations/fourier_processing/fp_cpu.hpp>
#include <libumpalumpa/operations/correlation/correlation_cpu.hpp>
#include <libumpalumpa/operations/extrema_finder/single_extrema_finder_cpu.hpp>

using umpalumpa::data::ManagedBy;
using namespace umpalumpa;

template<typename T>
FlexAlignCPU<T>::FlexAlignCPU()
  : forwardFFTOp(std::make_unique<fourier_transformation::FFTCPU>()),
    inverseFFTOp(std::make_unique<fourier_transformation::FFTCPU>()),
    cropOp(std::make_unique<fourier_processing::FPCPU>()),
    corrOp(std::make_unique<correlation::Correlation_CPU>()),
    extremaFinderOp(std::make_unique<extrema_finder::SingleExtremaFinderCPU>())
{}

template<typename T>
PhysicalDescriptor FlexAlignCPU<T>::CreatePD(size_t bytes, DataType type, bool copyInRAM, bool)
{
  void *ptr = nullptr;
  if (0 != bytes) {
    ptr = calloc(bytes, 1);
  }
  return PhysicalDescriptor(ptr, bytes, type, ManagedBy::Manually, nullptr);
}

template<typename T> void FlexAlignCPU<T>::RemovePD(const PhysicalDescriptor &pd)
{
  free(pd.GetPtr());
}

template class FlexAlignCPU<float>;