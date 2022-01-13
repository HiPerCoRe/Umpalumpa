#pragma once

#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <tests/algorithms/common.hpp>
#include <tests/utils.hpp>

using namespace umpalumpa::fourier_transformation;
using namespace umpalumpa::data;
using namespace umpalumpa::test;

class FFT_Tests : public TestAlg<AFFT>
{
protected:
  auto CreatePayloadSpatial(const Size &size)
  {
    auto ld = FourierDescriptor(size);
    auto type = DataType::Get<float>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Spatial data");
  }

  auto CreatePayloadFrequencyOut(const Settings &settings, const Size &size)
  {
    auto fd = FourierDescriptor::FourierSpaceDescriptor{};
    auto ld = FourierDescriptor(size, PaddingDescriptor(), fd);
    auto type = DataType::Get<std::complex<float>>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = [settings, this, bytes, type]() {
      if (settings.IsOutOfPlace()) {
        return Create(bytes, type);
      } else {
        // TODO find out if in StarPU we can use the same Physical Descriptor
        // or if we have to create a new handle
        throw std::runtime_error("Not implemented yet!");
      }
    }();
    return Payload(ld, std::move(pd), "Output data");
  }

  void SetUp(const Settings &settings, const Size &size)
  {
    pSpatial = std::make_unique<Payload<FourierDescriptor>>(CreatePayloadSpatial(size));
    Register(pSpatial->dataInfo);

    pFrequency =
      std::make_unique<Payload<FourierDescriptor>>(CreatePayloadFrequencyOut(settings, size));
    Register(pFrequency->dataInfo);
  }

  /**
   * Called at the end of each test fixture
   **/
  void TearDown() override
  {
    Unregister(pSpatial->dataInfo);
    Remove(pSpatial->dataInfo);

    Unregister(pFrequency->dataInfo);
    Remove(pFrequency->dataInfo);
  }

  template<typename T, typename P, typename F> void Generate(P &p, F f)
  {
    Acquire(p.dataInfo);
    auto *inData = reinterpret_cast<T *>(p.GetPtr());
    memset(inData, 0, p.dataInfo.GetBytes());
    f(inData);
    // PrintData(inData, p.info.GetSize());
    Release(p.dataInfo);
  }

  void testFFTInpulseOrigin(AFFT::OutputData &out, AFFT::InputData &in, const Settings &settings)
  {
    Generate<float>(in.GetData(), [&in](auto *ptr) {
      for (size_t n = 0; n < in.GetData().info.GetSize().n; ++n) {
        // impulse at the origin ...
        ptr[n * in.GetData().info.GetPaddedSize().single] = 1.f;
      }
    });

    auto &alg = GetAlg();

    ASSERT_TRUE(alg.Init(out, in, settings));
    ASSERT_TRUE(alg.Execute(out, in));
    // wait till the work is done
    alg.Synchronize();
    // make sure that data are on this memory node
    Acquire(out.GetData().dataInfo);
    // check results
    auto *outData = reinterpret_cast<std::complex<float> *>(out.GetData().GetPtr());
    // PrintData(outData, out.GetData().info.GetPaddedSize());

    float delta = 0.00001f;
    for (size_t i = 0; i < out.GetData().info.GetPaddedSize().total; ++i) {
      // ... will result in constant real value, and no imag value
      ASSERT_NEAR(1, outData[i].real(), delta) << " at " << i;
      ASSERT_NEAR(0, outData[i].imag(), delta) << " at " << i;
    }
    // we're done with those data
    Release(out.GetData().dataInfo);
  }

  void testIFFTInpulseOrigin(AFFT::OutputData &out, AFFT::InputData &in, const Settings &settings)
  {
    Generate<std::complex<float>>(in.GetData(), [&in](auto *ptr) {
      for (size_t n = 0; n < in.GetData().info.GetPaddedSize().single; ++n) {
        // constant real value, and no imag value ...
        ptr[n] = { 1.f, 0 };
      }
    });

    auto &alg = GetAlg();

    ASSERT_TRUE(alg.Init(out, in, settings));
    ASSERT_TRUE(alg.Execute(out, in));
    // wait till the work is done
    alg.Synchronize();
    // make sure that data are on this memory node
    Acquire(out.GetData().dataInfo);
    // check results
    auto *outData = reinterpret_cast<float *>(out.GetData().GetPtr());
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
    // we're done with those data
    Release(out.GetData().dataInfo);
  }

  void testFFTInpulseShifted(AFFT::OutputData &out, AFFT::InputData &in, const Settings &settings)
  {
    Generate<float>(in.GetData(), [&in](auto *ptr) {
      for (size_t n = 0; n < in.GetData().info.GetSize().n; ++n) {
        // impulse at the origin ...
        ptr[n * in.GetData().info.GetPaddedSize().single + 1] = 1.f;
      }
    });

    auto &alg = GetAlg();

    ASSERT_TRUE(alg.Init(out, in, settings));
    ASSERT_TRUE(alg.Execute(out, in));
    // wait till the work is done
    alg.Synchronize();
    // make sure that data are on this memory node
    Acquire(out.GetData().dataInfo);
    // check results
    auto *outData = reinterpret_cast<std::complex<float> *>(out.GetData().GetPtr());
    // PrintData(outData, out.GetData().info.GetPaddedSize());

    float delta = 0.00001f;
    for (size_t i = 0; i < out.GetData().info.GetPaddedSize().total; ++i) {
      // ... will result in constant magnitude
      auto re = outData[i].real();
      auto im = outData[i].imag();
      auto mag = re * re + im * im;
      ASSERT_NEAR(1.f, std::sqrt(mag), delta) << " at " << i;
    }
    // we're done with those data
    Release(out.GetData().dataInfo);
  }

  void testFFTIFFT(AFFT::OutputData &out,
    AFFT::InputData &in,
    const Settings &settings,
    size_t batchSize = 0)
  {
    auto *outData = reinterpret_cast<std::complex<float> *>(out.GetData().GetPtr());
    auto inverseIn = AFFT::InputData(out.GetData());
    auto inverseOut = AFFT::OutputData(in.GetData());

    // generate reference data
    auto ref = std::make_unique<float[]>(in.GetData().info.GetPaddedSize().total);
    // FillRandom(ref, in.GetData().dataInfo.GetBytes());
    FillNormalDist(ref.get(), in.GetData().info.GetPaddedSize().total);

    Acquire(in.GetData().dataInfo);
    auto *inData = reinterpret_cast<float *>(in.GetData().GetPtr());
    memcpy(
      in.GetData().GetPtr(), ref.get(), in.GetData().info.GetPaddedSize().total * sizeof(float));
    // Print(inData, in.GetData().info.GetPaddedSize());
    Release(in.GetData().dataInfo);

    auto &alg = GetAlg();
    auto forwardSettings =
      (settings.GetDirection() == Direction::kForward) ? settings : settings.CreateInverse();

    if (0 == batchSize) {
      ASSERT_TRUE(alg.Init(out, in, forwardSettings));
      ASSERT_TRUE(alg.Execute(out, in));
    } else {
      GTEST_SKIP() << "Batching is not yet supported";
      // for (size_t offset = 0; offset < in.GetData().info.GetSize().n; offset += batchSize) {
      //   auto tmpOut = AFFT::OutputData(out.GetData().Subset(offset, batchSize));
      //   auto tmpIn = AFFT::InputData(in.GetData().Subset(offset, batchSize));
      //   if (0 == offset) { ASSERT_TRUE(alg.Init(tmpOut, tmpIn, forwardSettings)); }
      //   ASSERT_TRUE(alg.Execute(tmpOut, tmpIn));
      // }
    }

    auto inverseSettings = forwardSettings.CreateInverse();
    if (0 == batchSize) {
      ASSERT_TRUE(alg.Init(inverseOut, inverseIn, inverseSettings));
      ASSERT_TRUE(alg.Execute(inverseOut, inverseIn));
    } else {
      GTEST_SKIP() << "Batching is not yet supported";
      // for (size_t offset = 0; offset < in.GetData().info.GetSize().n; offset += batchSize) {
      //   auto tmpOut = AFFT::OutputData(inverseOut.GetData().Subset(offset, batchSize));
      //   auto tmpIn = AFFT::InputData(inverseIn.GetData().Subset(offset, batchSize));
      //   if (0 == offset) { ASSERT_TRUE(alg.Init(tmpOut, tmpIn, inverseSettings)); }
      //   ASSERT_TRUE(alg.Execute(tmpOut, tmpIn));
      // }
    }
    // wait till the work is done
    alg.Synchronize();
    // make sure that data are on this memory node
    Acquire(inverseOut.GetData().dataInfo);
    // check results
    // Print(inData, in.GetData().info.GetPaddedSize());

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
            ref.get() + offset,
            inverseOut.GetData().info.GetSize().x,
            [&ref, offset, &inData, normFact, delta](auto &pos) {
              auto dist = std::distance(ref.get() + offset, &pos);
              EXPECT_NEAR(pos, inData[offset + dist] * normFact, delta) << "at " << dist;
            });
        }
      }
    }
    // we're done with those data
    Release(inverseOut.GetData().dataInfo);
  }

  std::unique_ptr<Payload<FourierDescriptor>> pSpatial;
  std::unique_ptr<Payload<FourierDescriptor>> pFrequency;
};