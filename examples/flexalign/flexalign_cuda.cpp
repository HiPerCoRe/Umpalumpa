#include "flexalign_cuda.hpp"
#include <cassert>
#include <iostream>

template<typename T>
std::unique_ptr<Payload<FourierDescriptor>> FlexAlignCUDA<T>::ConvertToFFTAndCrop(size_t index,
  Payload<LogicalDescriptor> &img,
  const Size &cropSize)
{
  std::cout << "[CUDA]: FFT and crop of image " << index << "\n";
  auto ld = FourierDescriptor(cropSize);
  return std::make_unique<Payload<FourierDescriptor>>(ld, "");
};

template class FlexAlignCUDA<float>;