#include <fcntl.h>
#include <gtest/gtest.h>
#include <libumpalumpa/algorithms/fourier_transformation/settings.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <complex>
#include <random>
#include <execution>

using namespace umpalumpa::fourier_transformation;
using namespace umpalumpa::data;

// FIXME make some utility file, for similar functions needed in tests
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
  std::for_each(
    std::execution::par_unseq, data, data + elems, [&dist, &mt](auto &&e) { e = dist(mt); });
}

class FFT_Tests
{
public:
  virtual AFFT &GetTransformer() = 0;
  virtual void *Allocate(size_t bytes) = 0;
  virtual void Free(void *ptr) = 0;

  void testFFTInpulseOrigin(AFFT::OutputData &out, AFFT::InputData &in, const Settings &settings)
  {
    auto *inData = reinterpret_cast<float *>(in.GetData().GetPtr());
    auto *outData = reinterpret_cast<std::complex<float> *>(out.GetData().GetPtr());

    for (size_t n = 0; n < in.GetData().info.GetSize().n; ++n) {
      // impulse at the origin ...
      inData[n * in.GetData().info.GetPaddedSize().single] = 1.f;
    }

    // PrintData(inData, in.GetData().info.GetSize());

    auto &ft = GetTransformer();

    ASSERT_TRUE(ft.Init(out, in, settings));
    ASSERT_TRUE(ft.Execute(out, in));
    ft.Synchronize();

    // PrintData(outData, out.GetData().info.GetPaddedSize());

    float delta = 0.00001f;
    for (size_t i = 0; i < out.GetData().info.GetPaddedSize().total; ++i) {
      // ... will result in constant real value, and no imag value
      ASSERT_NEAR(1, outData[i].real(), delta) << " at " << i;
      ASSERT_NEAR(0, outData[i].imag(), delta) << " at " << i;
    }
  }

  void testIFFTInpulseOrigin(AFFT::OutputData &out, AFFT::InputData &in, const Settings &settings)
  {
    auto *inData = reinterpret_cast<std::complex<float> *>(in.GetData().GetPtr());
    auto *outData = reinterpret_cast<float *>(out.GetData().GetPtr());

    for (size_t n = 0; n < in.GetData().info.GetPaddedSize().single; ++n) {
      // constant real value, and no imag value ...
      inData[n] = { 1.f, 0 };
    }

    // PrintData(inData, in.GetData().info.GetPaddedSize());

    auto &ft = GetTransformer();

    ASSERT_TRUE(ft.Init(out, in, settings));
    ASSERT_TRUE(ft.Execute(out, in));
    ft.Synchronize();

    // PrintData(outData, out.GetData().info.GetSize());

    float delta = 0.00001f;
    for (size_t n = 0; n < out.GetData().info.GetSize().n; ++n) {
      size_t offset = n * out.GetData().info.GetPaddedSize().single;
      // skip the padded area, it can contain garbage data
      for (size_t z = 0; z < out.GetData().info.GetSize().z; ++z) {
        for (size_t y = 0; y < out.GetData().info.GetSize().y; ++y) {
          for (size_t x = 0; x < out.GetData().info.GetSize().x; ++x) {
            size_t index = offset
                           + z * out.GetData().info.GetSize().x * out.GetData().info.GetSize().y
                           + y * out.GetData().info.GetSize().x + x;
            // output is not normalized, so normalize it to make the the test more stable
            if (index == offset) {
              // ... will result in impulse at the origin ...
              ASSERT_NEAR(1.f, outData[index] / out.GetData().info.GetSize().single, delta)
                << "at " << index;
            } else {
              // ... and zeros elsewhere
              ASSERT_NEAR(0.f, outData[index] / out.GetData().info.GetSize().single, delta)
                << "at " << index;
            }
          }
        }
      }
    }
  }

  void testFFTInpulseShifted(AFFT::OutputData &out, AFFT::InputData &in, const Settings &settings)
  {
    auto *inData = reinterpret_cast<float *>(in.GetData().GetPtr());
    auto *outData = reinterpret_cast<std::complex<float> *>(out.GetData().GetPtr());

    for (size_t n = 0; n < in.GetData().info.GetSize().n; ++n) {
      // impulse at the origin ...
      inData[n * in.GetData().info.GetPaddedSize().single + 1] = 1.f;
    }

    // PrintData(inData, in.GetData().info.GetSize());

    auto &ft = GetTransformer();

    ASSERT_TRUE(ft.Init(out, in, settings));
    ASSERT_TRUE(ft.Execute(out, in));
    ft.Synchronize();

    // PrintData(outData, out.GetData().info.GetPaddedSize());

    float delta = 0.00001f;
    for (size_t i = 0; i < out.GetData().info.GetPaddedSize().total; ++i) {
      // ... will result in constant magnitude
      auto re = outData[i].real();
      auto im = outData[i].imag();
      auto mag = re * re + im * im;
      ASSERT_NEAR(1.f, std::sqrt(mag), delta) << " at " << i;
    }
  }

  void testFFTIFFT(AFFT::OutputData &out,
    AFFT::InputData &in,
    const Settings &settings,
    size_t batchSize = 0)
  {
    auto *inData = reinterpret_cast<float *>(in.GetData().GetPtr());
    auto *outData = reinterpret_cast<std::complex<float> *>(out.GetData().GetPtr());
    auto inverseIn = AFFT::InputData(out.GetData());
    auto inverseOut = AFFT::OutputData(in.GetData());

    auto *ref = new float[in.GetData().info.GetPaddedSize().total];
    // FillRandomBytes(ref, in.GetData().dataInfo.GetBytes());
    GenerateData(ref, in.GetData().info.GetPaddedSize().total);
    memcpy(in.GetData().GetPtr(), ref, in.GetData().info.GetPaddedSize().total * sizeof(float));

    // PrintData(inData, in.GetData().info.GetPaddedSize());

    auto &ft = GetTransformer();
    auto forwardSettings =
      (settings.GetDirection() == Direction::kForward) ? settings : settings.CreateInverse();

    if (0 == batchSize) {
      ASSERT_TRUE(ft.Init(out, in, forwardSettings));
      ASSERT_TRUE(ft.Execute(out, in));
    } else {
      for (size_t offset = 0; offset < in.GetData().info.GetSize().n; offset += batchSize) {
        auto tmpOut = AFFT::OutputData(out.GetData().Subset(offset, batchSize));
        auto tmpIn = AFFT::InputData(in.GetData().Subset(offset, batchSize));
        if (0 == offset) { ASSERT_TRUE(ft.Init(tmpOut, tmpIn, forwardSettings)); }
        ASSERT_TRUE(ft.Execute(tmpOut, tmpIn));
      }
    }
    ft.Synchronize();

    // PrintData(outData, out.GetData().info.GetPaddedSize());

    auto inverseSettings = forwardSettings.CreateInverse();
    if (0 == batchSize) {
      ASSERT_TRUE(ft.Init(inverseOut, inverseIn, inverseSettings));
      ASSERT_TRUE(ft.Execute(inverseOut, inverseIn));
    } else {
      for (size_t offset = 0; offset < in.GetData().info.GetSize().n; offset += batchSize) {
        auto tmpOut = AFFT::OutputData(inverseOut.GetData().Subset(offset, batchSize));
        auto tmpIn = AFFT::InputData(inverseIn.GetData().Subset(offset, batchSize));
        if (0 == offset) { ASSERT_TRUE(ft.Init(tmpOut, tmpIn, inverseSettings)); }
        ASSERT_TRUE(ft.Execute(tmpOut, tmpIn));
      }
    }
    ft.Synchronize();

    // PrintData(inData, in.GetData().info.GetPaddedSize());

    float delta = 0.00001f;
    const float normFact = inverseOut.GetData().info.GetNormFactor();
    for (size_t n = 0; n < inverseOut.GetData().info.GetSize().n; ++n) {
      size_t offsetN = n * inverseOut.GetData().info.GetPaddedSize().single;
      // skip the padded area, it can contain garbage data
      for (size_t z = 0; z < inverseOut.GetData().info.GetSize().z; ++z) {
        for (size_t y = 0; y < inverseOut.GetData().info.GetSize().y; ++y) {
          auto offset =
            offsetN
            + z * inverseOut.GetData().info.GetSize().x * inverseOut.GetData().info.GetSize().y
            + y * inverseOut.GetData().info.GetSize().x;

          std::for_each_n(std::execution::par,
            ref + offset,
            inverseOut.GetData().info.GetSize().x,
            [&ref, offset, &inData, normFact, delta](auto &pos) {
              auto dist = std::distance(ref + offset, &pos);
              ASSERT_NEAR(pos, inData[offset + dist] * normFact, delta) << "at " << dist;
            });
        }
      }
    }

    delete[] ref;
  }

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
  void SetUpFFT(const Settings &settings, const Size &size, const PaddingDescriptor &padding)
  {
    ldSpatial = std::make_unique<FourierDescriptor>(size, padding);

    auto spatialSizeInBytes = ldSpatial->GetPaddedSize().total * Sizeof(DataType::kFloat);
    dataSpatial = std::shared_ptr<void>(Allocate(spatialSizeInBytes), GetFree());
    memset(dataSpatial.get(), 0, spatialSizeInBytes);
    pdSpatial =
      std::make_unique<PhysicalDescriptor>(dataSpatial.get(), spatialSizeInBytes, DataType::kFloat);

    ldFrequency = std::make_unique<FourierDescriptor>(
      size, padding, FourierDescriptor::FourierSpaceDescriptor());

    auto frequencySizeInBytes = ldFrequency->Elems() * Sizeof(DataType::kComplexFloat);
    if (settings.IsOutOfPlace()) {
      dataFrequency = std::shared_ptr<void>(Allocate(frequencySizeInBytes), GetFree());
    } else {
      dataFrequency = dataSpatial;
    }
    pdFrequency = std::make_unique<PhysicalDescriptor>(
      dataFrequency.get(), frequencySizeInBytes, DataType::kComplexFloat);
  }

  std::shared_ptr<void> dataSpatial;
  std::unique_ptr<PhysicalDescriptor> pdSpatial;
  std::unique_ptr<FourierDescriptor> ldSpatial;
  std::shared_ptr<void> dataFrequency;
  std::unique_ptr<PhysicalDescriptor> pdFrequency;
  std::unique_ptr<FourierDescriptor> ldFrequency;
};
