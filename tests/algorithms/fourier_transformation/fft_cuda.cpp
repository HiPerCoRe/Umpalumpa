#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <gtest/gtest.h>
#include <libumpalumpa/utils/cuda.hpp>
#include <tests/algorithms/fourier_transformation/fft_tests.hpp>

#include <cuda_runtime.h>

using namespace umpalumpa::fourier_transformation;
using namespace umpalumpa::data;

class FFTCUDATest : public ::testing::Test, public FFT_Tests
{
public:
  static auto Allocate(size_t bytes)
  {
    void *ptr;
    CudaErrchk(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  static void Free(void *ptr) { cudaFree(ptr); }

  // Deliberately not using gtest's SetUp method, because we need to know Settings and
  // Size of the current test
  // ONLY float currently supported!!
  auto SetUpFFT(const Settings &settings, const Size &size, const Size &paddedSize) {
    ldSpatial = std::make_unique<FourierDescriptor>(size, paddedSize);
    auto spatialSizeInBytes = ldSpatial->GetPaddedSize().total * Sizeof(DataType::kFloat);
    pdSpatial = std::make_unique<PhysicalDescriptor>(spatialSizeInBytes, DataType::kFloat);

    dataSpatial = std::shared_ptr<void>(Allocate(pdSpatial->bytes), FFTCUDATest::Free);
    memset(dataSpatial.get(), 0, pdSpatial->bytes);

    ldFrequency = std::make_unique<FourierDescriptor>(size, paddedSize, FourierDescriptor::FourierSpaceDescriptor());
    auto frequencySizeInBytes = ldFrequency->GetPaddedSize().total * Sizeof(DataType::kFloat) * 2;
    pdFrequency = std::make_unique<PhysicalDescriptor>(frequencySizeInBytes, DataType::kFloat);

    if (settings.IsOutOfPlace()) {
      dataFrequency = std::shared_ptr<void>(Allocate(pdFrequency->bytes), FFTCUDATest::Free);
    } else {
      dataFrequency = dataSpatial;
    }
  }

  FFTCUDA &GetTransformer() override { return transformer; }

protected:
  FFTCUDA transformer = FFTCUDA(0);

  std::shared_ptr<void> dataSpatial;
  std::unique_ptr<PhysicalDescriptor> pdSpatial;
  std::unique_ptr<FourierDescriptor> ldSpatial;
  std::shared_ptr<void> dataFrequency;
  std::unique_ptr<PhysicalDescriptor> pdFrequency;
  std::unique_ptr<FourierDescriptor> ldFrequency;
};
#define NAME FFTCUDATest
#include <tests/algorithms/fourier_transformation/afft_common.hpp>

