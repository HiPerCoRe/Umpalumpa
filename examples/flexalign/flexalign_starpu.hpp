#pragma once

#include "flexalign.hpp"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

/**
 * This example implements StarPU implementation of FlexAlign
 **/
template<typename T> class FlexAlignStarPU : public FlexAlign<T>
{
public:
  FlexAlignStarPU();
  virtual ~FlexAlignStarPU();

protected:
  PhysicalDescriptor CreatePD(size_t bytes, DataType type, bool copyInRAM, bool pinned) override;

  void RemovePD(const PhysicalDescriptor &pd) override;

  AFFT &GetForwardFFTOp() const override { return *forwardFFTOp; }

  AFFT &GetInverseFFTOp() const override { return *inverseFFTOp; }

  AFP &GetCropOp() const override { return *cropOp; }

  ACorrelation &GetCorrelationOp() const override { return *corrOp; }

  AExtremaFinder &GetFindMaxOp() const override { return *extremaFinderOp; }

  void Acquire(const PhysicalDescriptor &p) const override;

  void Release(const PhysicalDescriptor &p) const override;

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

  std::unique_ptr<AFFT> forwardFFTOp;
  std::unique_ptr<AFFT> inverseFFTOp;
  std::unique_ptr<AFP> cropOp;
  std::unique_ptr<ACorrelation> corrOp;
  std::unique_ptr<AExtremaFinder> extremaFinderOp;

  void RemoveFromQueue();
  std::mutex mutex;
  std::queue<PDData> toRemove;
  std::condition_variable workAvailable;
  std::unique_ptr<std::thread> thr;
  bool keepWorking = true;
};