#include "fr_cuda.hpp"
#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cuda.hpp>

using umpalumpa::data::ManagedBy;
using namespace umpalumpa;

template<typename T>
FourierReconstructionCUDA<T>::FourierReconstructionCUDA()
  : FFTAlg(std::make_unique<fourier_transformation::FFTCUDA>(worker)),
    cropAlg(std::make_unique<fourier_processing::FPCUDA>(worker)),
    FRAlg(std::make_unique<fourier_reconstruction::FRCUDA>(worker))
{}

template<typename T>
PhysicalDescriptor
  FourierReconstructionCUDA<T>::CreatePD(size_t bytes, DataType type, bool copyInRAM, bool pinned)
{
  void *ptr = nullptr;
  if (0 != bytes) { CudaErrchk(cudaMallocManaged(&ptr, bytes)); }
  auto pd = PhysicalDescriptor(ptr, bytes, type, ManagedBy::CUDA, nullptr);
  pd.SetPinned(pinned);
  return pd;
}

template<typename T> void FourierReconstructionCUDA<T>::RemovePD(const PhysicalDescriptor &pd)
{
  CudaErrchk(cudaFree(pd.GetPtr()));
}

template<typename T> void FourierReconstructionCUDA<T>::Acquire(const PhysicalDescriptor &pd) const
{
  CudaErrchk(cudaMemPrefetchAsync(pd.GetPtr(), pd.GetBytes(), worker));
}

template class FourierReconstructionCUDA<float>;