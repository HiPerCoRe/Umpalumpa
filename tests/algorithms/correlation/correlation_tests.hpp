#include <fcntl.h>
#include <gtest/gtest.h>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <libumpalumpa/algorithms/correlation/acorrelation.hpp>
#include <complex>
#include <random>

using namespace umpalumpa::correlation;
using namespace umpalumpa::data;

class Correlation_Tests 
{
public:
  virtual ACorrelation &GetTransformer() = 0;
  virtual void *Allocate(size_t bytes) = 0;
  virtual void Free(void *ptr) = 0;

  // Works correctly only with float
  void testCorrelationSimple(ACorrelation::OutputData &out, ACorrelation::InputData &in, const Settings &settings) {
    auto f = [](std::complex<float> *arr, size_t size) {
        for (size_t i = 0; i < size; i++) {
          arr[i] = {1.f, 1.f};
        }
      };
    testCorrelation(out, in, settings, f);
  }

  // Works correctly only with float
  void testCorrelationRandomData(ACorrelation::OutputData &out, ACorrelation::InputData &in, const Settings &settings) {
    // Not part of the function f, so that more invocations produce different output
    auto mt = std::mt19937(42);
    auto dist = std::normal_distribution<float>((float)0, (float)1);
    auto f = [&mt, &dist](std::complex<float> *arr, size_t size) {
        for (size_t i = 0; i < size; i++) {
          arr[i] = {dist(mt), dist(mt)};
        }
      };
    testCorrelation(out, in, settings, f);
  }

protected:
  template<typename InputProvider>
  void testCorrelation(ACorrelation::OutputData &out, ACorrelation::InputData &in, const Settings &settings, InputProvider ip) {
    auto *input1 = reinterpret_cast<std::complex<float>*>(in.data1.ptr);
    auto *input2 = reinterpret_cast<std::complex<float>*>(in.data2.ptr);
    auto *output = reinterpret_cast<std::complex<float>*>(out.data.ptr);
    auto inSize = in.data1.info.GetSize();
    auto inSize2 = in.data2.info.GetSize();
    //auto outSize = out.data.info.GetSize();

    ip(input1, inSize.total);
    if (input1 != input2) {
      ip(input2, inSize2.total);
    }

    //PrintData(input1, inSize);

    auto &corr = GetTransformer();

    corr.Init(out, in, settings);
    corr.Execute(out, in);
    corr.Synchronize();

    //PrintData(output, outSize);

    float delta = 0.00001f;
    check(output, settings, input1, inSize, input2, inSize2, delta);
  }

  void check(const std::complex<float> *output, const Settings &s, const std::complex<float> *input1,
      const Size &inSize, const std::complex<float> *input2, const Size &in2Size, float delta = 0.00001f) const {
    auto imageCounter = 0;
    // all pairs (combination without repetitions), in reasonable ordering
    for (size_t i1 = 0; i1 < inSize.n; i1++) {
      for (size_t i2 = input1 == input2 ? i1 + 1 : 0; i2 < in2Size.n; i2++) {
        auto imageOffset = imageCounter * inSize.single;
        // all pixels (2D)
        for (size_t y = 0; y < inSize.y; y++) {
          for (size_t x = 0; x < inSize.x; x++) {
            auto pixelOffset = y * inSize.x + x;
            auto totalOffset = imageOffset + pixelOffset;
            auto val1 = input1[i1 * inSize.single + y * inSize.x + x];
            auto val2 = input2[i2 * inSize.single + y * inSize.x + x];
            auto realRes = val1.real()*val2.real() + val1.imag()*val2.imag();
            auto imagRes = val1.imag()*val2.real() - val1.real()*val2.imag();
            if (s.GetCenter()) {
              float centerCoef = 1 - 2*((static_cast<int>(x)+static_cast<int>(y))&1);
              realRes *= centerCoef;
              imagRes *= centerCoef;
            }
            ASSERT_NEAR(realRes, output[totalOffset].real(), delta) << " at <" << i1 << "," << i2 << "> (" << x << "," << y << ")";
            ASSERT_NEAR(imagRes, output[totalOffset].imag(), delta) << " at <" << i1 << "," << i2 << "> (" << x << "," << y << ")";
          }
        }
        imageCounter++;
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
  // Assumes size == paddedSize
  void SetUpCorrelation(const Settings &settings, const Size &inSize, int in2N = 0) {
    bool isWithin = in2N == 0;
    size_t outImages = inSize.n * in2N;
    if (isWithin) {
      in2N = inSize.n;
      for (size_t i = 0; i < inSize.n; i++)
        for (size_t j = i + 1; j < inSize.n; j++)
          outImages++;
    }

    auto in2Size = Size(inSize.x, inSize.y, inSize.z, in2N);
    auto outSize = Size(inSize.x, inSize.y, inSize.z, outImages);

    ldIn1 = std::make_unique<FourierDescriptor>(inSize, PaddingDescriptor(), FourierDescriptor::FourierSpaceDescriptor{});
    auto inputSizeInBytes = ldIn1->GetPaddedSize().total * Sizeof(DataType::kComplexFloat);
    pdIn1 = std::make_unique<PhysicalDescriptor>(inputSizeInBytes, DataType::kComplexFloat);

    inData1 = std::shared_ptr<void>(Allocate(pdIn1->bytes), GetFree());
    memset(inData1.get(), 0, pdIn1->bytes);

    ldIn2 = std::make_unique<FourierDescriptor>(in2Size, PaddingDescriptor(), FourierDescriptor::FourierSpaceDescriptor{});
    auto input2SizeInBytes = ldIn2->GetPaddedSize().total * Sizeof(DataType::kComplexFloat);
    pdIn2 = std::make_unique<PhysicalDescriptor>(input2SizeInBytes, DataType::kComplexFloat);

    if (isWithin) {
      inData2 = inData1;
    } else {
      inData2 = std::shared_ptr<void>(Allocate(pdIn2->bytes), GetFree());
      memset(inData2.get(), 0, pdIn2->bytes);
    }

    ldOut = std::make_unique<FourierDescriptor>(outSize, PaddingDescriptor(), FourierDescriptor::FourierSpaceDescriptor{});
    auto outputSizeInBytes = ldOut->GetPaddedSize().total * Sizeof(DataType::kComplexFloat);
    pdOut = std::make_unique<PhysicalDescriptor>(outputSizeInBytes, DataType::kComplexFloat);

    outData = std::shared_ptr<void>(Allocate(pdOut->bytes), GetFree());
  }

  std::shared_ptr<void> inData1;
  std::unique_ptr<PhysicalDescriptor> pdIn1;
  std::unique_ptr<FourierDescriptor> ldIn1;
  std::shared_ptr<void> inData2;
  std::unique_ptr<PhysicalDescriptor> pdIn2;
  std::unique_ptr<FourierDescriptor> ldIn2;
  std::shared_ptr<void> outData;
  std::unique_ptr<PhysicalDescriptor> pdOut;
  std::unique_ptr<FourierDescriptor> ldOut;
};
