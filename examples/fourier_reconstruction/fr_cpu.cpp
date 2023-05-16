#include "fr_cpu.hpp"
#include <libumpalumpa/operations/fourier_transformation/fft_cpu.hpp>
#include <libumpalumpa/operations/fourier_processing/fp_cpu.hpp>
#include <libumpalumpa/operations/fourier_reconstruction/fr_cpu.hpp>

using umpalumpa::data::ManagedBy;
using namespace umpalumpa;

template<typename T>
FourierReconstructionCPU<T>::FourierReconstructionCPU()
  : FFTOp(std::make_unique<fourier_transformation::FFTCPU>()),
    cropOp(std::make_unique<fourier_processing::FPCPU>()),
    FROp(std::make_unique<fourier_reconstruction::FRCPU>())
{}

template<typename T>
PhysicalDescriptor
  FourierReconstructionCPU<T>::CreatePD(size_t bytes, DataType type, bool copyInRAM, bool pinned)
{
  void *ptr = nullptr;
  if (0 != bytes) { ptr = calloc(bytes, 1); }
  // FIXME pin data
  auto pd = PhysicalDescriptor(ptr, bytes, type, ManagedBy::Manually, nullptr);
  pd.SetPinned(pinned);
  return pd;
}

template<typename T> void FourierReconstructionCPU<T>::RemovePD(const PhysicalDescriptor &pd)
{
  free(pd.GetPtr());
}

template class FourierReconstructionCPU<float>;