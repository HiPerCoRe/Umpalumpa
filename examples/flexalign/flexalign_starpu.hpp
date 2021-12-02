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

  AFFT &GetForwardFFTAlg() const override { return *forwardFFTAlg; }

  AFFT &GetInverseFFTAlg() const override { return *inverseFFTAlg; }

  AFP &GetCropAlg() const override { return *cropAlg; }

  ACorrelation &GetCorrelationAlg() const override { return *corrAlg; }

  AExtremaFinder &GetFindMaxAlg() const override { return *extremaFinderAlg;}

  void Acquire(const PhysicalDescriptor &p) const override;

  void Release(const PhysicalDescriptor &p) const override;


private:
  std::unique_ptr<AFFT> forwardFFTAlg;
  std::unique_ptr<AFFT> inverseFFTAlg;
  std::unique_ptr<AFP> cropAlg;
  std::unique_ptr<ACorrelation> corrAlg;
  std::unique_ptr<AExtremaFinder> extremaFinderAlg;
};