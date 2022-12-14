#pragma once

#include "flexalign.hpp"

/**
 * This example implements CPU-only implementation of FlexAlign
 **/
template<typename T> class FlexAlignCPU : public FlexAlign<T>
{
public:
  FlexAlignCPU();
  virtual ~FlexAlignCPU() = default;

protected:
  PhysicalDescriptor CreatePD(size_t bytes, DataType type, bool copyInRAM, bool pinned) override;

  void RemovePD(const PhysicalDescriptor &pd) override;

  AFFT &GetForwardFFTAlg() const override { return *forwardFFTAlg; }

  AFFT &GetInverseFFTAlg() const override { return *inverseFFTAlg; }

  AFP &GetCropAlg() const override { return *cropAlg; }

  ACorrelation &GetCorrelationAlg() const override { return *corrAlg; }

  AExtremaFinder &GetFindMaxAlg() const override { return *extremaFinderAlg; }

  void Acquire(const PhysicalDescriptor &p) const override{ /* nothing to do */ };

  void Release(const PhysicalDescriptor &p) const override{ /* nothing to do */ };

private:
  std::unique_ptr<AFFT> forwardFFTAlg;
  std::unique_ptr<AFFT> inverseFFTAlg;
  std::unique_ptr<AFP> cropAlg;
  std::unique_ptr<ACorrelation> corrAlg;
  std::unique_ptr<AExtremaFinder> extremaFinderAlg;
};