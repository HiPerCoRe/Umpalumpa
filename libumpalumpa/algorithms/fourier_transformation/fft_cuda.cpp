#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <libumpalumpa/system_includes/cufftXt.hpp>
#include <array>
#include <stdexcept>

namespace umpalumpa::fourier_transformation {

template<typename F> void FFTCUDA::manyHelper(F function)
{
  const auto &in = this->GetInputRef();
  auto &fd = in.GetData().info;
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
    type = (in.GetData().dataInfo.GetType().Is<float>()) ? CUFFT_R2C : CUFFT_D2Z;
  } else {
    idist = static_cast<int>(fd.GetPaddedFrequencySize().single);
    odist = static_cast<int>(fd.GetPaddedSpatialSize().single);
    type = (in.GetData().dataInfo.GetType().Is<std::complex<float>>()) ? CUFFT_C2R : CUFFT_Z2D;
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

void FFTCUDA::Cleanup()
{
  if (IsInitialized()) { CudaErrchk(cufftDestroy(plan)); }
  AFFT::Cleanup();
}

bool FFTCUDA::InitImpl()
{
  CudaErrchk(cufftCreate(&plan));
  setupPlan();

  return true;
}

bool FFTCUDA::ExecuteImpl(const OutputData &out, const InputData &in)
{
  auto direction = (GetSettings().IsForward() ? CUFFT_FORWARD : CUFFT_INVERSE);
  CudaErrchk(cufftXtExec(plan, in.GetData().GetPtr(), out.GetData().GetPtr(), direction));
  return true;
}

bool FFTCUDA::IsValid(const OutputData &out, const InputData &in, const Settings &s) const
{
  // Too many elements for Fourier Transformation. It would cause int overflow in the cuda kernel
  return AFFT::IsValid(out, in, s)
         && in.GetData().info.GetPaddedSpatialSize().total
              <= static_cast<size_t>(std::numeric_limits<int>::max());
}

void FFTCUDA::setupPlan()
{
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
    CudaErrchk(
      cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch));
  };
  manyHelper(f);
  CudaErrchk(cufftSetStream(plan, stream));
}

FFTCUDA::FFTCUDA(int deviceOrdinal) : shouldDestroyStream(true)
{
  CudaErrchk(cuInit(0));
  CUdevice device;
  CudaErrchk(cuDeviceGet(&device, deviceOrdinal));
  CUcontext context;// FIXME test that stream is created on the correct device
  cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
  CudaErrchk(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
}

FFTCUDA::~FFTCUDA()
{
  this->Cleanup();
  if (shouldDestroyStream) {
    // stream will be deleted once there's no more work on it
    CudaErrchk(cuStreamDestroy(stream));
  }
}

void FFTCUDA::Synchronize() { CudaErrchk(cudaStreamSynchronize(stream)); }

size_t FFTCUDA::GetUsedBytes() const
{
  size_t result = 0;
  if (this->IsInitialized()) { CudaErrchk(cufftGetSize(plan, &result)); }
  return result;
}
}// namespace umpalumpa::fourier_transformation
