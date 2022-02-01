#include "fr_cpu.hpp"
#include <libumpalumpa/algorithms/fourier_transformation/fft_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cpu.hpp>

using umpalumpa::data::ManagedBy;
using namespace umpalumpa;

template<typename T>
FourierReconstructionCPU<T>::FourierReconstructionCPU()
  : FFTAlg(std::make_unique<fourier_transformation::FFTCPU>()),
    cropAlg(std::make_unique<fourier_processing::FPCPU>()),
    FRAlg(std::make_unique<fourier_reconstruction::FRCPU>())
{}

template<typename T>
PhysicalDescriptor
  FourierReconstructionCPU<T>::CreatePD(size_t bytes, DataType type, bool copyInRAM, bool)
{
  void *ptr = nullptr;
  if (0 != bytes) { ptr = calloc(bytes, 1); }
  return PhysicalDescriptor(ptr, bytes, type, ManagedBy::Manually, nullptr);
}

template<typename T> void FourierReconstructionCPU<T>::RemovePD(const PhysicalDescriptor &pd, bool)
{
  free(pd.GetPtr());
}

template class FourierReconstructionCPU<float>;