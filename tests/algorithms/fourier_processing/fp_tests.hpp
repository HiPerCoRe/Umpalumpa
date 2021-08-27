#include <fcntl.h>
#include <gtest/gtest.h>
#include <libumpalumpa/algorithms/fourier_transformation/settings.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>
#include <complex>
#include <random>

using namespace umpalumpa::fourier_processing;
using namespace umpalumpa::data;

//FIXME make some utility file, for similar functions needed in tests
template<typename T> void FillRandomBytes(T *dst, size_t bytes)
{
  int fd = open("/dev/urandom", O_RDONLY);
  read(fd, dst, bytes);
  close(fd);
}

template<typename T> void GenerateData(T *data, size_t elems)
{
  auto mt = std::mt19937(42);
  auto dist = std::normal_distribution<float>((float)0, (float)1);
  for (size_t i = 0; i < elems; ++i) { data[i] = dist(mt); }
}

class FP_Tests 
{
public:
  virtual AFP &GetTransformer() = 0;
  virtual void *Allocate(size_t bytes) = 0;
  virtual void Free(void *ptr) = 0;

  // TESTS GO HERE

protected:
  template<typename T> void PrintData(T *data, const Size size)
  {
    ASSERT_EQ(size.GetDim(), Dimensionality::k2Dim);
    for (size_t n = 0; n < size.n; ++n) {
      size_t offset = n * size.single;
      for (size_t y = 0; y < size.y; ++y) {
        for (size_t x = 0; x < size.x; ++x) { printf("%+.3f\t", data[offset + y * size.x + x]); }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }

  void PrintData(std::complex<float> *data, const Size size)
  {
    ASSERT_EQ(size.GetDim(), Dimensionality::k2Dim);
    for (size_t n = 0; n < size.n; ++n) {
      size_t offset = n * size.single;
      for (size_t y = 0; y < size.y; ++y) {
        for (size_t x = 0; x < size.x; ++x) {
          auto v = data[offset + y * size.x + x];
          printf("(%+.3f,%+.3f)\t", v.real(), v.imag() );
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }

  using FreeFunction = std::function<void(void*)>;

  // Could be left as pure virtual, but in case of an error, it might be
  // difficult to realize, where the pure virtual method was called. Instead
  // throw reasonable exception.
  virtual FreeFunction GetFree() {
    throw std::logic_error("GetFree() method not overridden!");
  }

  // Deliberately not using gtest's SetUp method, because we need to know Settings and
  // Size of the current test to properly initialize memory
  // ONLY float currently supported!!
  void SetUpFFT(const Settings &settings, const Size &size, const Size &paddedSize) {
    ldSpatial = std::make_unique<FourierDescriptor>(size, paddedSize);
    auto spatialSizeInBytes = ldSpatial->GetPaddedSize().total * Sizeof(DataType::kFloat);
    pdSpatial = std::make_unique<PhysicalDescriptor>(spatialSizeInBytes, DataType::kFloat);

    dataSpatial = std::shared_ptr<void>(Allocate(pdSpatial->bytes), GetFree());
    memset(dataSpatial.get(), 0, pdSpatial->bytes);

    ldFrequency = std::make_unique<FourierDescriptor>(size, paddedSize, FourierDescriptor::FourierSpaceDescriptor());
    auto frequencySizeInBytes = ldFrequency->GetPaddedSize().total * Sizeof(DataType::kFloat) * 2;
    pdFrequency = std::make_unique<PhysicalDescriptor>(frequencySizeInBytes, DataType::kFloat);

    if (settings.IsOutOfPlace()) {
      dataFrequency = std::shared_ptr<void>(Allocate(pdFrequency->bytes), GetFree());
    } else {
      dataFrequency = dataSpatial;
    }
  }

  std::shared_ptr<void> dataSpatial;
  std::unique_ptr<PhysicalDescriptor> pdSpatial;
  std::unique_ptr<FourierDescriptor> ldSpatial;
  std::shared_ptr<void> dataFrequency;
  std::unique_ptr<PhysicalDescriptor> pdFrequency;
  std::unique_ptr<FourierDescriptor> ldFrequency;
};
