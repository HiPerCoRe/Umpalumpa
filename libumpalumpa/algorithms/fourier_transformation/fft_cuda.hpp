#pragma once
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/system_includes/cufftXt.hpp>
#include <libumpalumpa/utils/cuda.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>
#include <array>
#include <stdexcept>
#include <libumpalumpa/data/data_type.hpp>

namespace umpalumpa {
namespace fourier_transformation {
  class FFTCUDA : public AFFT
  {
  public:
    bool Init(const ResultData &out, const InputData &in, const Settings &settings) override {
      SetSettings(settings);
      CudaErrchk(cufftDestroy(plan));

      cufftCreate(&plan);

      return true;
    }

    bool Execute(const ResultData &out, const InputData &in) override {
      // TODO create methods for comparing this InputData with Init InputData
      auto direction = (GetSettings().IsForward() ? CUFFT_FORWARD : CUFFT_INVERSE);
      CudaErrchk(cufftXtExec(plan, in.data.data, out.data.data, direction));
      return true;
    }

    void Cleanup() override {
      CudaErrchk(cufftDestroy(plan));
    }

  private:
    cufftHandle plan;

    //TODO move definitions into .cpp file
    template<typename F>
    void manyHelper(const InputData &in, const Settings &settings, F function) {
      auto &fd = in.data.info;
      auto n = std::array<int, 3>{
        static_cast<int>(fd.paddedSize.z),
        static_cast<int>(fd.paddedSize.y),
        static_cast<int>(fd.paddedSize.x)};
      int idist;
      int odist;
      cufftType type;

      if (settings.IsForward()) {
        idist = fd.paddedSize.single;
        odist = fd.frequencyDomainSize.single;
        // We know that data type is either float or double (validated before)
        type = (in.data.dataInfo.type == data::DataType::kFloat) ? CUFFT_R2C : CUFFT_D2Z;
      } else {
        idist = fd.frequencyDomainSizePadded.single;
        odist = fd.paddedSize.single;
        type = (in.data.dataInfo.type == data::DataType::kFloat) ? CUFFT_C2R : CUFFT_Z2D;
      }

      int rank = ToInt(fd.paddedSize.getDim());
      int offset = 3 - rank;

      function(rank, &n[offset], nullptr,
          1, idist, nullptr, 1, odist, type, fd.paddedSize.n);
    }

    void createPlan(const InputData &in, const Settings &settings) {
      if (in.data.info.paddedSize.total > std::numeric_limits<int>::max()) {
        throw std::length_error("Too many elements for Fourier Transformation. "
            "It would cause int overflow in the cuda kernel. Try to decrease batch size");
      }

      auto f = [&] (int rank, int *n, int *inembed,
          int istride, int idist, int *onembed, int ostride,
          int odist, cufftType type, int batch) {
        CudaErrchk(cufftPlanMany(&plan, rank, n, inembed,
              istride, idist, onembed, ostride,
              odist, type, batch));
      };
      manyHelper(in, settings, f);
      //CudaErrchk(cufftSetStream(plan, *(cudaStream_t)gpu.stream()));
    }

  };
}
}
