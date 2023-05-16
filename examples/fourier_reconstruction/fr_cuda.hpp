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

  void RemovePD(const PhysicalDescriptor &pd) override;

  AFFT &GetFFTOp() const override { return *FFTOp; }

  AFP &GetCropOp() const override { return *cropOp; }

  AFR &GetFROp() const override { return *FROp; }

  void Acquire(const PhysicalDescriptor &p) const override;

  void Release(const PhysicalDescriptor &p) const override{ /* nothing to do */ };

private:
  const int worker = 0;

  std::unique_ptr<AFFT> FFTOp;
  std::unique_ptr<AFP> cropOp;
  std::unique_ptr<AFR> FROp;
};