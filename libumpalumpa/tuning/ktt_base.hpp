#pragma once

#include <libumpalumpa/tuning/ktt_provider.hpp>
#include <cassert>

typedef struct CUstream_st *CUstream;

namespace umpalumpa::tuning {
class KTT_Base
{
public:
  /**
   * Recommended contructor. Pass streams that you want to use.
   * See documentation of the specific operation to know how many streams you should pass.
   * Notice that workerId != CUDA device ordinal.
   * CUDA device is detected from the first stream.
   **/
  // FIXME why do we pass the workerID? KTT device is detected from the stream
  explicit KTT_Base(int workerId, const std::vector<CUstream> &s) : streams(s), KTTId(workerId)
  {
    this->EnsureStreams();
    KTTProvider::Ensure(KTTId, CreateComputeQueues());
  }

  /**
   * Contructor which will also create all required streams.
   * Notice that workerId != CUDA device ordinal, though this constructor
   * will use it that way.
   **/
  explicit KTT_Base(int workerId) : streams(), KTTId(workerId)
  {
    this->CreateStreams();
    KTTProvider::Ensure(KTTId, CreateComputeQueues());
  }

  virtual ~KTT_Base() = default;// FIXME release streams if necessary

  KTTHelper &GetHelper() const { return KTTProvider::Get(KTTId); }

protected:
  virtual unsigned GetNoOfStreams() const { return 1; }

private:
  void EnsureStreams() const { assert(GetNoOfStreams() == streams.size()); }
  void CreateStreams();
  std::vector<ktt::ComputeQueue> CreateComputeQueues() const;

  std::vector<CUstream> streams;
  int KTTId;
};
}// namespace umpalumpa::tuning
