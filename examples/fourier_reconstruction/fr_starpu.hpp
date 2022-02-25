#pragma once

#include "fr.hpp"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

/**
 * This example implements StarPU implementation of the Fourier Reconstruction
 **/
template<typename T> class FourierReconstructionStarPU : public FourierReconstruction<T>
{
public:
  FourierReconstructionStarPU();
  virtual ~FourierReconstructionStarPU();

protected:
  PhysicalDescriptor CreatePD(size_t bytes, DataType type, bool copyInRAM, bool pinned) override;

  void RemovePD(const PhysicalDescriptor &pd) override;

  AFFT &GetFFTAlg() const override { return *FFTAlg; }

  AFP &GetCropAlg() const override { return *cropAlg; }

  AFR &GetFRAlg() const override { return *FRAlg; }

  void Acquire(const PhysicalDescriptor &p) const override;

  void Release(const PhysicalDescriptor &p) const override;

  void OptionalSynch() override;

private:
  struct PDData
  {
    PDData(const PhysicalDescriptor &pd)
      : handle(pd.GetHandle()), ptr(pd.GetPtr()), bytes(pd.GetBytes()), isPinned(pd.IsPinned())
    {}
    void *handle;
    void *ptr;
    const size_t bytes;
    const bool isPinned;
  };

  static void SetAvailableBytesRAM();
  static void SetAvailableBytesCUDA();

  void RemoveFromQueue();

  std::unique_ptr<AFFT> FFTAlg;
  std::unique_ptr<AFP> cropAlg;
  std::unique_ptr<AFR> FRAlg;

  std::mutex mutex;
  std::queue<PDData> toRemove;
  std::condition_variable workAvailable;
  std::unique_ptr<std::thread> thr;
  bool keepWorking = true;
};