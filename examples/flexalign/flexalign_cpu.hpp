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

  AFFT &GetForwardFFTOp() const override { return *forwardFFTOp; }

  AFFT &GetInverseFFTOp() const override { return *inverseFFTOp; }

  AFP &GetCropOp() const override { return *cropOp; }

  ACorrelation &GetCorrelationOp() const override { return *corrOp; }

  AExtremaFinder &GetFindMaxOp() const override { return *extremaFinderOp; }

  void Acquire(const PhysicalDescriptor &p) const override{ /* nothing to do */ };

  void Release(const PhysicalDescriptor &p) const override{ /* nothing to do */ };

private:
  std::unique_ptr<AFFT> forwardFFTOp;
  std::unique_ptr<AFFT> inverseFFTOp;
  std::unique_ptr<AFP> cropOp;
  std::unique_ptr<ACorrelation> corrOp;
  std::unique_ptr<AExtremaFinder> extremaFinderOp;
};