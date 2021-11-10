#include "flexalign_cpu.hpp"
#include <cassert>
#include <iostream>

template<typename T>
std::unique_ptr<Payload<FourierDescriptor>> FlexAlignCPU<T>::ConvertToFFTAndCrop(size_t index,
  Payload<LogicalDescriptor> &img,
  const Size &cropSize)
{
  std::cout << "[CPU]: FFT and crop of image " << index << "\n";
  auto ld = FourierDescriptor(cropSize);
  return std::make_unique<Payload<FourierDescriptor>>(ld, "");
};

template class FlexAlignCPU<float>;