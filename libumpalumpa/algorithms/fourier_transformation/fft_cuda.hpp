#pragma once
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/system_includes/cufftXt.hpp>
#include <libumpalumpa/utils/cuda.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>
#include <array>
#include <vector>
#include <stdexcept>

namespace umpalumpa {
namespace fourier_transformation {
  class FFTCUDA : public AFFT
  {
  public:
    /**
     * Constructor to create Fast Fourier Transformer which uses specific GPU stream.
     * Due to compatibility with the rest of the code, this constructor expects a vector
     * of streams, however only first first stream (at position 0) will be used.
     **/
    explicit FFTCUDA(const std::vector<CUstream> &s) : plan(0), stream(s.at(0)) {}

    explicit FFTCUDA(int deviceOrdinal) : plan(0)
    {
      CudaErrchk(cuInit(0));
      CUdevice device;
      CudaErrchk(cuDeviceGet(&device, deviceOrdinal));
      CUcontext context;// FIXME test that stream is created on the correct device
      cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
      CudaErrchk(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    }

    bool Init(const OutputData &out, const InputData &in, const Settings &s) override
    {
      bool canProcess = checkTypes(out, in, s.GetDirection());
      if (!canProcess) return false;
      if (IsInitialized()) { CudaErrchk(cufftDestroy(plan)); }
      SetSettings(s);

      CudaErrchk(cufftCreate(&plan));
      setupPlan(in);

      return true;
    }

    bool Execute(const OutputData &out, const InputData &in) override
    {
      if (!this->IsInitialized()) return false;
      // TODO create methods for comparing this InputData with Init InputData
      auto direction = (GetSettings().IsForward() ? CUFFT_FORWARD : CUFFT_INVERSE);
      CudaErrchk(cufftXtExec(plan, in.data.ptr, out.data.ptr, direction));
      // CudaErrchk(cufftExecR2C(plan, (cufftReal*)in.data.ptr, (cufftComplex*)out.data.ptr));
      return true;
    }

    void Cleanup() override
    {
      CudaErrchk(cufftDestroy(plan));
      AFFT::Cleanup();
    }

    void Synchronize() override
    {
      CudaErrchk(cudaDeviceSynchronize());// FIXME synchronize the correct KTT stream
    }

  private:
    cufftHandle plan;
    CUstream stream;

    bool checkTypes(const OutputData &out, const InputData &in, Direction d)
    {
      if (Direction::kForward == d) {
        return ((out.data.dataInfo.type == data::DataType::kComplexFloat)
                 && (in.data.dataInfo.type == data::DataType::kFloat))
               || ((out.data.dataInfo.type == data::DataType::kComplexDouble)
                   && (in.data.dataInfo.type == data::DataType::kDouble));
      }
      return ((out.data.dataInfo.type == data::DataType::kFloat)
               && (in.data.dataInfo.type == data::DataType::kComplexFloat))
             || ((out.data.dataInfo.type == data::DataType::kDouble)
                 && (in.data.dataInfo.type == data::DataType::kComplexDouble));
    }

    // TODO move definitions into .cpp file
    template<typename F> void manyHelper(const InputData &in, F function)
    {
      auto &fd = in.data.info;
      auto n = std::array<int, 3>{ static_cast<int>(fd.GetPaddedSpatialSize().z),
        static_cast<int>(fd.GetPaddedSpatialSize().y),
        static_cast<int>(fd.GetPaddedSpatialSize().x) };
      int idist;
      int odist;
      cufftType type;

      if (GetSettings().IsForward()) {
        idist = static_cast<int>(fd.GetPaddedSpatialSize().single);
        odist = static_cast<int>(fd.GetFrequencySize().single);
        // We know that data type is either float or double (validated before)
        type = (in.data.dataInfo.type == data::DataType::kFloat) ? CUFFT_R2C : CUFFT_D2Z;
      } else {
        idist = static_cast<int>(fd.GetPaddedFrequencySize().single);
        odist = static_cast<int>(fd.GetPaddedSpatialSize().single);
        type = (in.data.dataInfo.type == data::DataType::kComplexFloat) ? CUFFT_C2R : CUFFT_Z2D;
      }

      int rank = ToInt(fd.GetPaddedSpatialSize().GetDim());
      size_t offset = 3 - static_cast<size_t>(rank);

      function(rank,
        &n[offset],
        nullptr,
        1,
        idist,
        nullptr,
        1,
        odist,
        type,
        static_cast<int>(fd.GetPaddedSpatialSize().n));
    }

    void setupPlan(const InputData &in)
    {
      if (in.data.info.GetPaddedSpatialSize().total > std::numeric_limits<int>::max()) {
        throw std::length_error(
          "Too many elements for Fourier Transformation. "
          "It would cause int overflow in the cuda kernel. Try to decrease batch size");
      }

      auto f = [&](int rank,
                 int *n,
                 int *inembed,
                 int istride,
                 int idist,
                 int *onembed,
                 int ostride,
                 int odist,
                 cufftType type,
                 int batch) {
        CudaErrchk(cufftPlanMany(
          &plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch));
      };
      manyHelper(in, f);
      // CudaErrchk(cufftSetStream(plan, *(cudaStream_t)gpu.stream()));
    }
  };
}// namespace fourier_transformation
}// namespace umpalumpa
