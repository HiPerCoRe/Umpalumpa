#pragma once

#include "fr.hpp"

/**
 * This example implements pure CUDA implementation of the Fourier Reconstruction
 **/
template<typename T> class FourierReconstructionCUDA : public FourierReconstruction<T>
{
public:
  FourierReconstructionCUDA();

protected:
  PhysicalDescriptor CreatePD(size_t bytes, DataType type, bool copyInRAM, bool pinned) override;

  void RemovePD(const PhysicalDescriptor &pd, bool pinned) override;

  AFFT &GetFFTAlg() const override { return *FFTAlg; }

  AFP &GetCropAlg() const override { return *cropAlg; }

  AFR &GetFRAlg() const override { return *FRAlg; }

  void Acquire(const PhysicalDescriptor &p) const override;

  void Release(const PhysicalDescriptor &p) const override{ /* nothing to do */ };

private:
  const int worker = 0;

  std::unique_ptr<AFFT> FFTAlg;
  std::unique_ptr<AFP> cropAlg;
  std::unique_ptr<AFR> FRAlg;
};