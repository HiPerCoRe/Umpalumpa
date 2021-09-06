#include <fcntl.h>
#include <gtest/gtest.h>
#include <libumpalumpa/algorithms/fourier_transformation/settings.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <complex>
#include <random>

using namespace umpalumpa::fourier_transformation;
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

class FFT_Tests 
{
public:
  virtual AFFT &GetTransformer() = 0;
  virtual void *Allocate(size_t bytes) = 0;
  virtual void Free(void *ptr) = 0;

  void testFFTInpulseOrigin(AFFT::ResultData &out, AFFT::InputData &in, const Settings &settings) {
    auto *inData = reinterpret_cast<float*>(in.data.ptr);
    auto *outData = reinterpret_cast<std::complex<float>*>(out.data.ptr);

    for (size_t n = 0; n < in.data.info.GetSize().n; ++n) {
      // impulse at the origin ...
      inData[n * in.data.info.GetPaddedSize().single] = 1.f;
    }

    PrintData(inData, in.data.info.GetSize());

    auto &ft = GetTransformer();

    ft.Init(out, in, settings);
    ft.Execute(out, in);
    ft.Synchronize();

    PrintData(outData, out.data.info.GetPaddedSize());

    float delta = 0.00001f;
    for (size_t i = 0; i < out.data.info.GetPaddedSize().total; ++i) {
      // ... will result in constant real value, and no imag value
      ASSERT_NEAR(1, outData[i].real(), delta) << " at " << i;
      ASSERT_NEAR(0, outData[i].imag(), delta) << " at " << i;
    }
  }

  void testIFFTInpulseOrigin(AFFT::ResultData &out, AFFT::InputData &in, const Settings &settings) {
    auto *inData = reinterpret_cast<std::complex<float>*>(in.data.ptr);
    auto *outData = reinterpret_cast<float*>(out.data.ptr);

    for (size_t n = 0; n < in.data.info.GetPaddedSize().single; ++n) {
      // impulse at the origin ...
      inData[n] = {1.f, 0};
    }

    PrintData(inData, in.data.info.GetPaddedSize());

    auto &ft = GetTransformer();

    ft.Init(out, in, settings);
    ft.Execute(out, in);
    ft.Synchronize();

    PrintData(outData, out.data.info.GetSize());

    float delta = 0.00001f;
    for (size_t n = 0; n < out.data.info.GetSize().n; ++n) {
      size_t offset = n * out.data.info.GetPaddedSize().single;
      // skip the padded area, it can contain garbage data
      for (size_t z = 0; z < out.data.info.GetSize().z; ++z) {
        for (size_t y = 0; y < out.data.info.GetSize().y; ++y) {
          for (size_t x = 0; x < out.data.info.GetSize().x; ++x) {
            size_t index = offset + z * out.data.info.GetSize().x * out.data.info.GetSize().y + y * out.data.info.GetSize().x + x;
            // output is not normalized, so normalize it to make the the test more stable
            if (index == offset) {
              // ... will result in impulse at the origin ...
              ASSERT_NEAR(1.f, outData[index] / out.data.info.GetSize().single, delta) << "at " << index;
            } else {
              // ... and zeros elsewhere
              ASSERT_NEAR(0.f, outData[index] / out.data.info.GetSize().single, delta) << "at " << index;
            }
          }
        }
      }
    }
  }

  void testFFTInpulseShifted(AFFT::ResultData &out, AFFT::InputData &in, const Settings &settings) {
    auto *inData = reinterpret_cast<float*>(in.data.ptr);
    auto *outData = reinterpret_cast<std::complex<float>*>(out.data.ptr);

    for (size_t n = 0; n < in.data.info.GetSize().n; ++n) {
      // impulse at the origin ...
      inData[n * in.data.info.GetPaddedSize().single + 1] = 1.f;
    }

    PrintData(inData, in.data.info.GetSize());

    auto &ft = GetTransformer();

    ft.Init(out, in, settings);
    ft.Execute(out, in);
    ft.Synchronize();

    PrintData(outData, out.data.info.GetPaddedSize());

    float delta = 0.00001f;
    for (size_t i = 0; i < out.data.info.GetPaddedSize().total; ++i) {
      // ... will result in constant magnitude
      auto re = outData[i].real();
      auto im = outData[i].imag();
      auto mag = re * re + im * im;
      ASSERT_NEAR(1.f, std::sqrt(mag), delta) << " at " << i;
    }
  }

  void testFFTIFFT(AFFT::ResultData &out, AFFT::InputData &in, const Settings &settings) {
    auto *inData = reinterpret_cast<float*>(in.data.ptr);
    auto *outData = reinterpret_cast<std::complex<float>*>(out.data.ptr);
    auto &inverseIn = out;
    auto &inverseOut = in;

    auto *ref = new float[in.data.info.GetPaddedSize().total];
    GenerateData(ref, in.data.info.GetPaddedSize().total);
    memcpy(in.data.ptr, ref, in.data.info.GetPaddedSize().total * sizeof(float));

    PrintData(inData, in.data.info.GetPaddedSize());

    auto &ft = GetTransformer();

    //FIXME we assume that settings is set as forward fft
    ft.Init(out, in, settings);
    ft.Execute(out, in);
    ft.Synchronize();

    PrintData(outData, out.data.info.GetPaddedSize());

    ft.Init(inverseOut, inverseIn, settings.CreateInverse());
    ft.Execute(inverseOut, inverseIn);
    ft.Synchronize();

    PrintData(inData, in.data.info.GetPaddedSize());

    float delta = 0.00001f;
    for (size_t n = 0; n < inverseOut.data.info.GetSize().n; ++n) {
      size_t offset = n * inverseOut.data.info.GetPaddedSize().single;
      // skip the padded area, it can contain garbage data
      for (size_t z = 0; z < inverseOut.data.info.GetSize().z; ++z) {
        for (size_t y = 0; y < inverseOut.data.info.GetSize().y; ++y) {
          for (size_t x = 0; x < inverseOut.data.info.GetSize().x; ++x) {
            size_t index = offset + z * inverseOut.data.info.GetSize().x * inverseOut.data.info.GetSize().y + y * inverseOut.data.info.GetSize().x + x;
            ASSERT_NEAR(ref[index], inData[index] / inverseOut.data.info.GetSize().single, delta) << "at " << index;
          }
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
