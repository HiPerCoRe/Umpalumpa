#pragma once

#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>
#include <tests/algorithms/common.hpp>
#include <tests/utils.hpp>

using namespace umpalumpa::fourier_processing;
using namespace umpalumpa::data;
using namespace umpalumpa::test;

/**
 * Class responsible for testing.
 * Specific implementation of the algorithms should inherit from it.
 **/
class FP_Tests : public TestAlg<AFP>
{
protected:
  auto CreatePayloadIn(const Settings &settings, const Size &size)
  {
    auto fd = FourierDescriptor::FourierSpaceDescriptor{};
    auto ld = FourierDescriptor(size, PaddingDescriptor(), fd);
    auto bytes = ld.Elems() * Sizeof(DataType::kComplexFloat);
    auto pd = Create(bytes, DataType::kComplexFloat);
    return Payload(ld, std::move(pd), "Input data");
  }

  auto CreatePayloadOut(const Settings &settings, const Size &size)
  {
    auto fd = FourierDescriptor::FourierSpaceDescriptor{};
    auto ld = FourierDescriptor(size, PaddingDescriptor(), fd);
    auto bytes = ld.Elems() * Sizeof(DataType::kComplexFloat);
    auto pd = [settings, this, bytes]() {
      if (settings.IsOutOfPlace()) {
        return Create(bytes, DataType::kComplexFloat);
      } else {
        // TODO find out if in StarPU we can use the same Physical Descriptor
        // or if we have to create a new handle
        throw std::runtime_error("Not implemented yet!");
      }
    }();
    return Payload(ld, std::move(pd), "Output data");
  }

  auto CreatePayloadFilter(const Settings &settings, const Size &size)
  {
    auto ld = LogicalDescriptor(size);
    auto bytes = ld.Elems() * Sizeof(DataType::kFloat);
    auto pd = Create(bytes, DataType::kFloat);
    return Payload(ld, std::move(pd), "Filter");
  }

  void SetUp(const Settings &settings, const Size &sizeIn, const Size &sizeOut)
  {
    pIn = std::make_unique<Payload<FourierDescriptor>>(CreatePayloadIn(settings, sizeIn));
    Register(pIn->dataInfo);

    pOut = std::make_unique<Payload<FourierDescriptor>>(CreatePayloadIn(settings, sizeOut));
    Register(pOut->dataInfo);

    pFilter = std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadFilter(settings, sizeOut));
    Register(pFilter->dataInfo);
  }

  /**
   * Called at the end of each test fixture
   **/
  void TearDown() override
  {
    Unregister(pIn->dataInfo);
    Remove(pIn->dataInfo);

    Unregister(pOut->dataInfo);
    Remove(pOut->dataInfo);

    Unregister(pFilter->dataInfo);
    Remove(pFilter->dataInfo);
  }

  std::unique_ptr<Payload<FourierDescriptor>> pIn;
  std::unique_ptr<Payload<FourierDescriptor>> pOut;
  std::unique_ptr<Payload<LogicalDescriptor>> pFilter;

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

    // Print(input, inSize);
    // Print(filter, in.GetFilter().info.GetSize());

    auto &alg = GetAlg();

    ASSERT_TRUE(alg.Init(out, in, settings));
    ASSERT_TRUE(alg.Execute(out, in));
    // wait till the work is done
    alg.Synchronize();
    // make sure that data are on this memory node
    Acquire(out.GetData().dataInfo);
    // check results
    float normFactor = 1.f / in.GetData().info.GetSpatialSize().single;
    check(output,
      outSize,
      input,
      inSize,
      in.GetData().info.GetSpatialSize(),
      normFactor,
      filter,
      settings);
    // we're done with those data
    Release(out.GetData().dataInfo);
  }

  void check(const std::complex<float> *output,
    const Size &outSize,
    const std::complex<float> *input,
    const Size &inSize,
    const Size &inSpatialSize,
    float normFactor,
    const float *filter,
    const Settings &s) const
  {
    for (size_t n = 0; n < outSize.n; n++) {
      size_t cropIndex = outSize.y / 2 + 1;
      for (size_t y = 0; y < outSize.y; y++) {
        for (size_t x = 0; x < outSize.x; x++) {
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
          if (s.GetMaxFreq().has_value()) {
            auto freq = [](float i, float max) { return ((i <= (max / 2)) ? i : (i - max)) / max; };
            auto max = s.GetMaxFreq().value();
            auto freqX = freq(x, inSpatialSize.x);
            auto freqY = freq(y, inSpatialSize.y);
            if ((freqX * freqX + freqY * freqY) > max) { inReal = inImag = 0; }
          }
          ASSERT_FLOAT_EQ(inReal, output[outIndex].real()) << " at real " << outIndex;
          ASSERT_FLOAT_EQ(inImag, output[outIndex].imag()) << " at imag " << outIndex;
        }
      }
    }
  }
};