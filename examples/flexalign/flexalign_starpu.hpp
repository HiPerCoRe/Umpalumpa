#pragma once

#include "flexalign.hpp"

/**
 * This example implements StarPU implementation of FlexAlign
 **/
template<typename T> class FlexAlignStarPU : public FlexAlign<T>
{
protected:
  std::unique_ptr<Payload<FourierDescriptor>> ConvertToFFTAndCrop(size_t index,
    Payload<LogicalDescriptor> &img,
    const Size &cropSize) override;
};