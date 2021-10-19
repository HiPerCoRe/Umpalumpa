#pragma once

#include <libumpalumpa/tuning/ktt_provider.hpp>
#include <cassert>

typedef struct CUstream_st *CUstream;

namespace umpalumpa {
namespace algorithm {
  class KTT_Base
  {
  public:
    /**
     * Recommended contructor. Pass streams that you want to use.
     * See documentation of the specific algorithm to know how many streams you should pass.
     * Notice that workerId != CUDA device ordinal.
     * CUDA device is detected from the first stream.
     **/
    explicit KTT_Base(int workerId, const std::vector<CUstream> &s) : streams(s), KTTId(workerId)
    {
      this->EnsureStreams();
      umpalumpa::utils::KTTProvider::Ensure(KTTId, CreateComputeQueues());
    }

    /**
     * Contructor which will also create all required streams.
     * Notice that workerId != CUDA device ordinal, though this constructor
     * will use it that way.
     **/
    explicit KTT_Base(int workerId) : streams(), KTTId(workerId)
    {
      this->CreateStreams();
      umpalumpa::utils::KTTProvider::Ensure(KTTId, CreateComputeQueues());
    }

    virtual ~KTT_Base() = default;

    utils::KTTHelper &GetHelper() const { return utils::KTTProvider::Get(KTTId); }

  protected:
    virtual unsigned GetNoOfStreams() const { return 1; }

  private:
    void EnsureStreams() const { assert(GetNoOfStreams() == streams.size()); }
    void CreateStreams();
    std::vector<ktt::ComputeQueue> CreateComputeQueues() const;

    std::vector<CUstream> streams;
    int KTTId;
  };
}// namespace algorithm
}// namespace umpalumpa
