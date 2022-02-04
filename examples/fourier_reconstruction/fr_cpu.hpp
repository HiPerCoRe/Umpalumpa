#pragma once

#include "fr.hpp"

/**
 * This example implements CPU-only implementation of Fourier Reconstruction
 **/
template<typename T> class FourierReconstructionCPU : public FourierReconstruction<T>
{
public:
  FourierReconstructionCPU();

protected:
  PhysicalDescriptor CreatePD(size_t bytes, DataType type, bool copyInRAM, bool pinned) override;

  void RemovePD(const PhysicalDescriptor &pd) override;

  AFFT &GetFFTAlg() const override { return *FFTAlg; }

  AFP &GetCropAlg() const override { return *cropAlg; }

  AFR &GetFRAlg() const override { return *FRAlg; }

  void Acquire(const PhysicalDescriptor &p) const override{ /* nothing to do */ };

  void Release(const PhysicalDescriptor &p) const override{ /* nothing to do */ };

private:
  std::unique_ptr<AFFT> FFTAlg;
  std::unique_ptr<AFP> cropAlg;
  std::unique_ptr<AFR> FRAlg;
};