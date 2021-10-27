#pragma once
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/utils/cuda.hpp>

namespace umpalumpa {
namespace fourier_transformation {
  class FFTCUDA final : public AFFT
  {
  public:
    /**
     * Constructor to create Fast Fourier Transformer which uses specific GPU stream.
     * Due to compatibility with the rest of the code, this constructor expects a vector
     * of streams, however only first first stream (at position 0) will be used.
     **/
    explicit FFTCUDA(const std::vector<CUstream> &s) : stream(s.at(0)) {}
    explicit FFTCUDA(int deviceOrdinal);
    ~FFTCUDA();
    void Synchronize() override;
    void Cleanup() override;

  protected:
    bool InitImpl() override;
    bool ExecuteImpl(const OutputData &out, const InputData &in);
    bool IsValid(const OutputData &out, const InputData &in, const Settings &s) override;

  private:
    template<typename F> void manyHelper(F function);
    void setupPlan();

    CUstream stream;
    cufftHandle plan;
    bool shouldDestroyStream = false;
  };
}// namespace fourier_transformation
}// namespace umpalumpa
