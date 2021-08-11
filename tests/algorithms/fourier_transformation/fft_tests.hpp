#include <gtest/gtest.h>
#include <libumpalumpa/algorithms/fourier_transformation/settings.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <complex>

using namespace umpalumpa::fourier_transformation;
using namespace umpalumpa::data;

class FFT_Tests 
{
  public:
    void testFFTInpulseOrigin(AFFT::ResultData &out, AFFT::InputData &in, const Settings &settings) {

      for (size_t n = 0; n < in.data.info.GetSize().n; ++n) {
        // impulse at the origin ...
        *(reinterpret_cast<float*>(in.data.data) + n * in.data.info.GetPaddedSize().single) = 1.f;
      }

      PrintData((float*)in.data.data, in.data.info.GetSize());

      auto &ft = GetTransformer();

      ft.Init(out, in, settings);
      ft.Execute(out, in);
      ft.Synchronize();

      PrintData((std::complex<float>*)out.data.data, out.data.info.GetPaddedSize());

      float delta = 0.00001f;
      for (size_t i = 0; i < out.data.info.GetPaddedSize().total; ++i) {
        // ... will result in constant real value, and no imag value
        auto *tmp = reinterpret_cast<std::complex<float>*>(out.data.data);
        ASSERT_NEAR(1, tmp[i].real(), delta) << " at " << i;
        ASSERT_NEAR(0, tmp[i].imag(), delta) << " at " << i;
      }
    }

    void testIFFTInpulseOrigin(AFFT::ResultData &out, AFFT::InputData &in, const Settings &settings) {

      for (size_t n = 0; n < in.data.info.GetPaddedSize().single; ++n) {
        // impulse at the origin ...
        *(reinterpret_cast<std::complex<float>*>(in.data.data) + n) = {1.f, 0};
      }

      PrintData((std::complex<float>*)in.data.data, in.data.info.GetPaddedSize());

      auto &ft = GetTransformer();

      ft.Init(out, in, settings);
      ft.Execute(out, in);
      ft.Synchronize();

      PrintData((float*)out.data.data, out.data.info.GetSize());

      float delta = 0.00001f;
      for (size_t n = 0; n < out.data.info.GetSize().n; ++n) {
        size_t offset = n * out.data.info.GetPaddedSize().single;
        // skip the padded area, it can contain garbage data
        for (size_t z = 0; z < out.data.info.GetSize().z; ++z) {
          for (size_t y = 0; y < out.data.info.GetSize().y; ++y) {
            for (size_t x = 0; x < out.data.info.GetSize().x; ++x) {
              size_t index = offset + z * out.data.info.GetSize().x * out.data.info.GetSize().y + y * out.data.info.GetSize().x + x;
              auto *tmp = reinterpret_cast<float*>(out.data.data);
              // output is not normalized, so normalize it to make the the test more stable
              if (index == offset) {
                // ... will result in impulse at the origin ...
                ASSERT_NEAR(1.f, tmp[index] / out.data.info.GetSize().single, delta) << "at " << index;
              } else {
                // ... and zeros elsewhere
                ASSERT_NEAR(0.f, tmp[index] / out.data.info.GetSize().single, delta) << "at " << index;
              }
            }
          }
        }
      }
    }

    void testFFTInpulseShifted(AFFT::ResultData &out, AFFT::InputData &in, const Settings &settings) {

      for (size_t n = 0; n < in.data.info.GetSize().n; ++n) {
        // impulse at the origin ...
        *(reinterpret_cast<float*>(in.data.data) + n * in.data.info.GetPaddedSize().single + 1) = 1.f;
      }

      PrintData((float*)in.data.data, in.data.info.GetSize());

      auto &ft = GetTransformer();

      ft.Init(out, in, settings);
      ft.Execute(out, in);
      ft.Synchronize();

      PrintData((std::complex<float>*)out.data.data, out.data.info.GetPaddedSize());

      float delta = 0.00001f;
      for (size_t i = 0; i < out.data.info.GetPaddedSize().total; ++i) {
        // ... will result in constant magnitude
        auto *tmp = reinterpret_cast<std::complex<float>*>(out.data.data);
        auto re = tmp[i].real();
        auto im = tmp[i].imag();
        auto mag = re * re + im * im;
        ASSERT_NEAR(1.f, std::sqrt(mag), delta) << " at " << i;
      }
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

    virtual AFFT &GetTransformer() = 0;
};
