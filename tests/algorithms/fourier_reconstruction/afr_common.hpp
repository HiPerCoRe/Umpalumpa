#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/afr.hpp>
#include <tests/algorithms/common.hpp>
#include <tests/utils.hpp>

using namespace umpalumpa::fourier_reconstruction;
using namespace umpalumpa::data;
using namespace umpalumpa::test;

/**
 * Class responsible for testing.
 * Specific implementation of the algorithms should inherit from it.
 **/
class FR_Tests : public TestAlg<AFR>
{
protected:
  auto CreatePayloadFFT(const Settings &settings, const Size &size)
  {
    auto fd = FourierDescriptor::FourierSpaceDescriptor{};
    auto ld = FourierDescriptor(size, PaddingDescriptor(), fd);
    auto bytes = ld.Elems() * Sizeof(DataType::kComplexFloat);
    auto pd = Create(bytes, DataType::kComplexFloat);
    return Payload(ld, std::move(pd), "Input projecttion data in FD");
  }

  auto CreatePayloadVolume(const Settings &settings, const Size &size)
  {
    auto fd = FourierDescriptor::FourierSpaceDescriptor{};
    auto ld = FourierDescriptor(size.CopyFor(1), PaddingDescriptor(), fd);
    auto bytes = ld.Elems() * Sizeof(DataType::kComplexFloat);
    auto pd = Create(bytes, DataType::kComplexFloat);
    return Payload(ld, std::move(pd), "Volume in FD");
  }

  auto CreatePayloadWeights(const Settings &settings, const Size &size)
  {
    auto ld = LogicalDescriptor(size);
    auto bytes = ld.Elems() * Sizeof(DataType::kFloat);
    auto pd = Create(bytes, DataType::kFloat);
    return Payload(ld, std::move(pd), "Weights");
  }

  void SetUp(const Settings &settings, const Size &projectionSize)
  {
    pFFT = std::make_unique<Payload<FourierDescriptor>>(CreatePayloadFFT(settings, projectionSize));
    Register(pFFT->dataInfo);

    auto volumeSize = Size(projectionSize.x, projectionSize.x, projectionSize.x, 1);
    pVolume =
      std::make_unique<Payload<FourierDescriptor>>(CreatePayloadVolume(settings, volumeSize));
    Register(pVolume->dataInfo);

    pWeight =
      std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadWeights(settings, volumeSize));
    Register(pWeight->dataInfo);
  }

  /**
   * Called at the end of each test fixture
   **/
  void TearDown() override
  {
    auto Clear = [this](auto &p) {
      Unregister(p->dataInfo);
      Remove(p->dataInfo);
    };

    Clear(pFFT);
    Clear(pVolume);
    Clear(pWeight);
  }

  std::unique_ptr<Payload<FourierDescriptor>> pFFT;
  std::unique_ptr<Payload<FourierDescriptor>> pVolume;
  std::unique_ptr<Payload<LogicalDescriptor>> pWeight;
};