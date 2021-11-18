#include "flexalign_starpu.hpp"
#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_starpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_starpu.hpp>

using umpalumpa::utils::StarPUUtils;
using umpalumpa::data::ManagedBy;
using namespace umpalumpa;


template<typename T>
FlexAlignStarPU<T>::FlexAlignStarPU()
  : fftAlg(std::make_unique<fourier_transformation::FFTStarPU>()),
    cropAlg(std::make_unique<fourier_processing::FPStarPU>())
{}
// template<typename T>
// Payload<FourierDescriptor> FlexAlignStarPU<T>::ConvertToFFTAndCrop(size_t index,
//   Payload<LogicalDescriptor> &img,
//   const Size &cropSize)
// {
//   std::cout << "[StarPU]: FFT and crop of image " << index << "\n";

// // define settings


//   auto cropSettings = fourier_processing::Settings(
//       fourier_transformation::Locality::kOutOfPlace);
//   cropSettings.SetApplyFilter(true);
//   cropSettings.SetNormalize(true);

//   auto ld = FourierDescriptor(cropSize);
//   auto type = this->GetComplexDataType();
//   auto bytes = ld.Elems() * Sizeof(type);
//   return Payload<FourierDescriptor>(ld, Create(bytes, type), "");
// };

template<typename T>
PhysicalDescriptor FlexAlignStarPU<T>::Create(size_t bytes, DataType type, bool tmp) const
{
  void *ptr = nullptr;
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
  StarPUUtils::Unregister(pd);
  delete StarPUUtils::GetHandle(pd);
  starpu_free(pd.GetPtr());
}

template class FlexAlignStarPU<float>;