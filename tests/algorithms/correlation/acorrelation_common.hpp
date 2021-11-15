#include <libumpalumpa/algorithms/correlation/acorrelation.hpp>
#include <tests/algorithms/common.hpp>
#include <tests/utils.hpp>
#include <optional>

using namespace umpalumpa::correlation;
using namespace umpalumpa::data;
using namespace umpalumpa::test;

class Correlation_Tests : public TestAlg<ACorrelation>
{
protected:
  virtual PhysicalDescriptor Copy(const PhysicalDescriptor &pd) = 0;

  auto CreatePayload(const Size &size, const std::string &name)
  {
    auto fd = FourierDescriptor::FourierSpaceDescriptor{};
    auto ld = FourierDescriptor(size, PaddingDescriptor(), fd);
    auto bytes = ld.Elems() * Sizeof(DataType::kComplexFloat);
    auto pd = Create(bytes, DataType::kComplexFloat);
    return Payload(ld, std::move(pd), name);
  }

  auto CreatePayloadData2(Payload<FourierDescriptor> const *p1, const Size &size)
  {
    if (p1) {
      // create new Payload pointing to the same data
      // be careful not to double-release resources of the p1
      // we need new handler, otherwise we might get deadlock when acquiring data
      return Payload(p1->info, Copy(p1->dataInfo), "Data 2");
    }
    auto fd = FourierDescriptor::FourierSpaceDescriptor{};
    auto ld = FourierDescriptor(size, PaddingDescriptor(), fd);
    auto bytes = ld.Elems() * Sizeof(DataType::kComplexFloat);
    auto pd = Create(bytes, DataType::kComplexFloat);
    return Payload(ld, std::move(pd), "Data 2");
  }

  void SetUp(const Settings &settings, const Size &size1, const Size &size2, bool isWithin)
  {
    auto nOut = isWithin ? ((size1.n * (size1.n - 1)) / 2) : (size1.n * size2.n);
    auto sizeOut = size1.CopyFor(nOut);
    pOut = std::make_unique<Payload<FourierDescriptor>>(CreatePayload(sizeOut, "Correlations"));
    Register(pOut->dataInfo);

    pData1 = std::make_unique<Payload<FourierDescriptor>>(CreatePayload(size1, "Data 1"));
    Register(pData1->dataInfo);

    pData2 = std::make_unique<Payload<FourierDescriptor>>(
      CreatePayloadData2(isWithin ? pData1.get() : nullptr, size2));
    Register(pData2->dataInfo);
  }

  /**
   * Called at the end of each test fixture
   **/
  void TearDown() override
  {
    // we should be able to release both Data, as they should be 'independent' Payloads
    Unregister(pData2->dataInfo);
    Remove(pData2->dataInfo);

    Unregister(pData1->dataInfo);
    Remove(pData1->dataInfo);

    Unregister(pOut->dataInfo);
    Remove(pOut->dataInfo);
  }

  // Works correctly only with float
  void testCorrelationSimple(ACorrelation::OutputData &out,
    ACorrelation::InputData &in,
    const Settings &settings)
  {
    auto f = [](std::complex<float> *arr, size_t size) {
      for (size_t i = 0; i < size; i++) { arr[i] = { 1.f, 1.f }; }
    };
    testCorrelation(out, in, settings, f);
  }

  // Works correctly only with float
  void testCorrelationRandomData(ACorrelation::OutputData &out,
    ACorrelation::InputData &in,
    const Settings &settings)
  {
    // Not part of the function f, so that more invocations produce different output
    auto mt = std::mt19937(42);
    auto dist = std::normal_distribution<float>((float)0, (float)1);
    auto f = [&mt, &dist](std::complex<float> *arr, size_t size) {
      for (size_t i = 0; i < size; i++) { arr[i] = { dist(mt), dist(mt) }; }
    };
    testCorrelation(out, in, settings, f);
  }

  template<typename InputProvider>
  void testCorrelation(ACorrelation::OutputData &out,
    ACorrelation::InputData &in,
    const Settings &settings,
    InputProvider ip)
  {
    auto *input1 = reinterpret_cast<std::complex<float> *>(in.GetData1().GetPtr());
    auto *input2 = reinterpret_cast<std::complex<float> *>(in.GetData2().GetPtr());
    auto inSize = in.GetData1().info.GetSize();
    auto inSize2 = in.GetData2().info.GetSize();

    Acquire(in.GetData1().dataInfo);
    Acquire(in.GetData2().dataInfo);
    ip(input1, inSize.total);
    if (input1 != input2) { ip(input2, inSize2.total); }
    // Print(input1, inSize);
    Release(in.GetData2().dataInfo);
    Release(in.GetData1().dataInfo);

    auto &corr = GetAlg();

    ASSERT_TRUE(corr.Init(out, in, settings));
    ASSERT_TRUE(corr.Execute(out, in));
    corr.Synchronize();

    // make sure that data are on this memory node
    Acquire(out.GetCorrelations().dataInfo);
    // check results
    auto *output = reinterpret_cast<std::complex<float> *>(out.GetCorrelations().GetPtr());
    auto outSize = out.GetCorrelations().info.GetSize();
    // Print(output, outSize);
    float delta = 0.00001f;
    check(output, settings, input1, inSize, input2, inSize2, delta);
    // we're done with those data
    Release(out.GetCorrelations().dataInfo);
  }

  void check(const std::complex<float> *output,
    const Settings &s,
    const std::complex<float> *input1,
    const Size &inSize,
    const std::complex<float> *input2,
    const Size &in2Size,
    float delta = 0.00001f) const
  {
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
            auto realRes = val1.real() * val2.real() + val1.imag() * val2.imag();
            auto imagRes = val1.imag() * val2.real() - val1.real() * val2.imag();
            if (s.GetCenter()) {
              float centerCoef = 1 - 2 * ((static_cast<int>(x) + static_cast<int>(y)) & 1);
              realRes *= centerCoef;
              imagRes *= centerCoef;
            }
            ASSERT_NEAR(realRes, output[totalOffset].real(), delta)
              << " at <" << i1 << "," << i2 << "> (" << x << "," << y << ")";
            ASSERT_NEAR(imagRes, output[totalOffset].imag(), delta)
              << " at <" << i1 << "," << i2 << "> (" << x << "," << y << ")";
          }
        }
        imageCounter++;
      }
    }
  }

  std::unique_ptr<Payload<FourierDescriptor>> pOut;
  std::unique_ptr<Payload<FourierDescriptor>> pData1;
  std::unique_ptr<Payload<FourierDescriptor>> pData2;
};
