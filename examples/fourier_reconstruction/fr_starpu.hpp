#pragma once

#include "fr.hpp"

/**
 * This example implements StarPU implementation of the Fourier Reconstruction
 **/
template<typename T> class FourierReconstructionStarPU : public FourierReconstruction<T>
{
public:
  FourierReconstructionStarPU();
  ~FourierReconstructionStarPU();

protected:
  PhysicalDescriptor CreatePD(size_t bytes, DataType type, bool copyInRAM, bool pinned) override;

  void RemovePD(const PhysicalDescriptor &pd, bool pinned) const override;

  AFFT &GetFFTAlg() const override { return *FFTAlg; }

  AFP &GetCropAlg() const override { return *cropAlg; }

  AFR &GetFRAlg() const override { return *FRAlg; }

  void Acquire(const PhysicalDescriptor &p) const override;

  void Release(const PhysicalDescriptor &p) const override;

private:
  static void SetAvailableBytesRAM();
  static void SetAvailableBytesCUDA();

  std::unique_ptr<AFFT> FFTAlg;
  std::unique_ptr<AFP> cropAlg;
  std::unique_ptr<AFR> FRAlg;
};