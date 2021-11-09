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

class FP_Tests
{
public:
  virtual AFP &GetFourierProcessor() = 0;
  virtual void *Allocate(size_t bytes) = 0;
  virtual void Free(void *ptr) = 0;
  virtual ManagedBy GetManager() = 0;
  virtual int GetMemoryNode() = 0;


  void testFP(AFP::OutputData &out, AFP::InputData &in, const Settings &settings)
  {
    auto *input = reinterpret_cast<std::complex<float> *>(in.GetData().GetPtr());
    auto *output = reinterpret_cast<std::complex<float> *>(out.GetData().GetPtr());
    auto *filter = reinterpret_cast<float *>(in.GetFilter().GetPtr());
    auto inSize = in.GetData().info.GetSize();
    auto outSize = out.GetData().info.GetSize();

    for (size_t i = 0; i < inSize.total; i++) {
      input[i] = { static_cast<float>(i), static_cast<float>(i) };
    }

    if (settings.GetApplyFilter()) {
      for (size_t i = 0; i < in.GetFilter().info.GetSize().total; i += 2) {
        reinterpret_cast<float *>(filter)[i] = -1.0f;
      }
      for (size_t i = 1; i < in.GetFilter().info.GetSize().total; i += 2) {
        reinterpret_cast<float *>(filter)[i] = 0.5f;
      }
    }

    // PrintData(input, inSize);
    // PrintData(filter, in.GetFilter().info.GetSize());

    auto &fp = GetFourierProcessor();

    ASSERT_TRUE(fp.Init(out, in, settings));
    ASSERT_TRUE(fp.Execute(out, in));
    fp.Synchronize();

    // PrintData(output, outSize);

    float delta = 0.00001f;
    checkEdges(output, outSize, delta);
    float normFactor = 1.f / in.GetData().info.GetSpatialSize().single;
    checkInside(output, outSize, input, inSize, normFactor, filter, settings, delta);
  }

protected:
  void checkEdges(const std::complex<float> *out, const Size &outSize, float delta = 0.00001f) const
  {
    for (size_t n = 0; n < outSize.n; n++) {
      for (size_t x = 0; x < outSize.x; x++) {
        auto outIndex = n * outSize.single + x;// y == 0
        ASSERT_NEAR(0.f, out[outIndex].real(), delta) << " at checkEdges";
        ASSERT_NEAR(0.f, out[outIndex].imag(), delta) << " at checkEdges";
      }
      for (size_t y = 0; y < outSize.y; y++) {
        auto outIndex = n * outSize.single + y * outSize.x;// x == 0
        ASSERT_NEAR(0.f, out[outIndex].real(), delta) << " at checkEdges";
        ASSERT_NEAR(0.f, out[outIndex].imag(), delta) << " at checkEdges";
      }
    }
  }

  void checkInside(const std::complex<float> *output,
    const Size &outSize,
    const std::complex<float> *input,
    const Size &inSize,
    float normFactor,
    const float *filter,
    const Settings &s,
    float delta = 0.00001f) const
  {
    for (size_t n = 0; n < outSize.n; n++) {
      size_t cropIndex = outSize.y / 2 + 1;
      for (size_t y = 1; y < outSize.y; y++) {
        for (size_t x = 1; x < outSize.x; x++) {
          auto inIndex =
            n * inSize.single + (y < cropIndex ? y : inSize.y - outSize.y + y) * inSize.x + x;
          auto outIndex = n * outSize.single + y * outSize.x + x;
          float inReal = input[inIndex].real();
          float inImag = input[inIndex].imag();
          if (s.GetApplyFilter()) {
            float filterCoef = filter[y * outSize.x + x];
            inReal *= filterCoef;
            inImag *= filterCoef;
          }
          if (s.GetNormalize()) {
            inReal *= normFactor;
            inImag *= normFactor;
          }
          if (s.GetCenter()) {
            float centerCoef =
              1 - 2 * ((static_cast<int>(x + y)) & 1);// center FT, input must be even
            inReal *= centerCoef;
            inImag *= centerCoef;
          }
          ASSERT_NEAR(inReal, output[outIndex].real(), delta) << " at real " << outIndex;
          ASSERT_NEAR(inImag, output[outIndex].imag(), delta) << " at imag " << outIndex;
        }
      }
    }
  }

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
          printf("(%+.3f,%+.3f)\t", v.real(), v.imag());
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }

  using FreeFunction = std::function<void(void *)>;

  // Could be left as pure virtual, but in case of an error, it might be
  // difficult to realize, where the pure virtual method was called. Instead
  // throw reasonable exception.
  virtual FreeFunction GetFree() { throw std::logic_error("GetFree() method not overridden!"); }

  // Deliberately not using gtest's SetUp method, because we need to know Settings and
  // Size of the current test to properly initialize memory
  // ONLY float currently supported!!
  // Assumes size == paddedSize
  void SetUpFP(const Settings &settings, const Size &inSize, const Size &outSize)
  {
    ldIn = std::make_unique<FourierDescriptor>(
      inSize, PaddingDescriptor(), FourierDescriptor::FourierSpaceDescriptor{});

    auto inputSizeInBytes = ldIn->Elems() * Sizeof(DataType::kComplexFloat);
    inData = std::shared_ptr<void>(Allocate(inputSizeInBytes), GetFree());
    memset(inData.get(), 0, inputSizeInBytes);
    pdIn = std::make_unique<PhysicalDescriptor>(
      inData.get(), inputSizeInBytes, DataType::kComplexFloat, GetManager(), GetMemoryNode());

    ldOut = std::make_unique<FourierDescriptor>(
      outSize, PaddingDescriptor(), FourierDescriptor::FourierSpaceDescriptor{});

    auto outputSizeInBytes = ldOut->Elems() * Sizeof(DataType::kComplexFloat);
    if (settings.IsOutOfPlace()) {
      outData = std::shared_ptr<void>(Allocate(outputSizeInBytes), GetFree());
    } else {
      outData = inData;
    }
    pdOut = std::make_unique<PhysicalDescriptor>(
      outData.get(), outputSizeInBytes, DataType::kComplexFloat, GetManager(), GetMemoryNode());

    ldFilter = std::make_unique<LogicalDescriptor>(outSize);

    auto filterSizeInBytes = ldFilter->Elems() * Sizeof(DataType::kFloat);
    filterData = std::shared_ptr<void>(Allocate(filterSizeInBytes), GetFree());
    memset(filterData.get(), 0, filterSizeInBytes);
    pdFilter = std::make_unique<PhysicalDescriptor>(
      filterData.get(), filterSizeInBytes, DataType::kFloat, GetManager(), GetMemoryNode());
  }

  std::shared_ptr<void> inData;
  std::unique_ptr<PhysicalDescriptor> pdIn;
  std::unique_ptr<FourierDescriptor> ldIn;
  std::shared_ptr<void> outData;
  std::unique_ptr<PhysicalDescriptor> pdOut;
  std::unique_ptr<FourierDescriptor> ldOut;
  std::shared_ptr<void> filterData;
  std::unique_ptr<PhysicalDescriptor> pdFilter;
  std::unique_ptr<LogicalDescriptor> ldFilter;
};
