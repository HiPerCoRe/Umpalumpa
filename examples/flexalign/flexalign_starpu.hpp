#pragma once

#include "flexalign.hpp"

/**
 * This example implements StarPU implementation of FlexAlign
 **/
template<typename T> class FlexAlignStarPU : public FlexAlign<T>
{
public:
  FlexAlignStarPU();

protected:
  // Payload<FourierDescriptor> ConvertToFFTAndCrop(size_t index,
  //   Payload<LogicalDescriptor> &img,
  //   const Size &cropSize) override;

  PhysicalDescriptor Create(size_t bytes, DataType type, bool tmp) const override;

  void Remove(const PhysicalDescriptor &pd) const override;

  AFFT &GetFFTAlg() const override { return *fftAlg; }

  AFP &GetCropAlg() const override { return *cropAlg; }

private:
  std::unique_ptr<AFFT> fftAlg;
  std::unique_ptr<AFP> cropAlg;
};