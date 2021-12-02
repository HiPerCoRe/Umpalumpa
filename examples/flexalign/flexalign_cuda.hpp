#pragma once

#include "flexalign.hpp"

/**
 * This example implements pure CUDA implementation of FlexAlign
 **/
template<typename T> class FlexAlignCUDA : public FlexAlign<T>
{
protected:
  std::unique_ptr<Payload<FourierDescriptor>> ConvertToFFTAndCrop(size_t index,
    Payload<LogicalDescriptor> &img,
    const Size &cropSize) override;
};