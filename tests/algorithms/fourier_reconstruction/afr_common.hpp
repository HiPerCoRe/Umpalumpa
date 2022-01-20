#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/afr.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space_generator.hpp>
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
    auto type = DataType::Get<std::complex<float>>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Input projecttion data in FD");
  }

  auto CreatePayloadVolume(const Settings &settings, const Size &size)
  {
    auto fd = FourierDescriptor::FourierSpaceDescriptor{};
    fd.hasSymetry = true;
    auto ld = FourierDescriptor(size.CopyFor(1), PaddingDescriptor(), fd);
    auto type = DataType::Get<std::complex<float>>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Volume in FD");
  }

  auto CreatePayloadWeights(const Settings &settings, const Size &size)
  {
    auto ld = LogicalDescriptor(size);
    auto type = DataType::Get<float>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Weights");
  }

  void FillTraverseSpace(const float transform[3][3],
    TraverseSpace &space,
    const Size &transformationSize,
    const Size &volumeSize,
    const Settings &s,
    float weight)
  {
    return computeTraverseSpace(
      transformationSize.y
        / 2,// FIXME this should be probably .x, but Xmipp implementation has it like this
      transformationSize.y,
      transform,
      space,
      volumeSize.x - 1,
      volumeSize.y - 1,
      s.GetType() == Settings::Type::kFast,
      s.GetBlobRadius(),
      weight);
  }

  auto CreatePayloadTraverseSpace(const Settings &settings)
  {
    // TODO pass number of spaces needed?
    auto ld = LogicalDescriptor(Size(1, 1, 1, 1));
    auto type = DataType::Get<TraverseSpace>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), "Traverse space");
  }

  void SetUp(const Settings &settings, const Size &projectionSize)
  {
    pFFT = std::make_unique<Payload<FourierDescriptor>>(CreatePayloadFFT(settings, projectionSize));
    Register(pFFT->dataInfo);

    // we need uniform cube in the fourier domain
    auto volumeSize = Size(projectionSize.y + 1, projectionSize.y + 1, projectionSize.y + 1, 1);
    pVolume =
      std::make_unique<Payload<FourierDescriptor>>(CreatePayloadVolume(settings, volumeSize));
    Register(pVolume->dataInfo);

    pWeight =
      std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadWeights(settings, volumeSize));
    Register(pWeight->dataInfo);

    pTraverseSpace =
      std::make_unique<Payload<LogicalDescriptor>>(CreatePayloadTraverseSpace(settings));
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
    Clear(pTraverseSpace);
  }

  std::unique_ptr<Payload<FourierDescriptor>> pFFT;
  std::unique_ptr<Payload<FourierDescriptor>> pVolume;
  std::unique_ptr<Payload<LogicalDescriptor>> pWeight;
  std::unique_ptr<Payload<LogicalDescriptor>> pTraverseSpace;
};