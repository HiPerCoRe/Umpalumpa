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
