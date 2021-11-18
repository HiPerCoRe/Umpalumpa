#include "flexalign.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <type_traits>
#include <libumpalumpa/utils/payload.hpp>

template<typename T> void FlexAlign<T>::Execute(const umpalumpa::data::Size &sizeAll)
{
  assert(sizeAll.x > 5);
  assert(sizeAll.y > 5);
  assert(sizeAll.z == 1);

  auto sizeSingle = sizeAll.CopyFor(1);
  auto sizeSingleCrop = Size(sizeSingle.x / 2, sizeSingle.y / 2, sizeSingle.z, 1);
  auto sizeCross = Size(3, 3, 1, 1);

  auto filter = CreatePayloadFilter(sizeSingleCrop);

  auto images = std::vector<Payload<LogicalDescriptor>>();
  images.reserve(sizeAll.n);
  auto ffts = std::vector<Payload<FourierDescriptor>>();
  ffts.reserve(sizeAll.n);

  auto imgCenter = Size(sizeSingle.x / 2, sizeSingle.y / 2, 1, 1);

  for (size_t j = 0; j < sizeAll.n; ++j) {
    images.emplace_back(CreatePayloadImage(j, sizeSingle));
    auto &img = images.at(j);
    GenerateClockArms(j, img, sizeCross, j, j);
    ffts.emplace_back(ConvertToFFTAndCrop(j, img, filter));
    for (size_t i = 0; i < j; ++i) {
      auto correlation = Correlate(i, j, ffts.at(i), ffts.at(j));
      auto shift = FindMax(i, j, correlation);
      std::cout << "Shift of img " << i << " and " << j << " is [" << shift.x << ", " << shift.y
                << "]\n";
    }
  }
  // Release allocated data
  for (const auto &p : ffts) { Remove(p.dataInfo); }
  for (const auto &p : images) { Remove(p.dataInfo); }
  Remove(filter.dataInfo);
}

template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::ConvertToFFTAndCrop(size_t index,
  Payload<LogicalDescriptor> &img,
  Payload<LogicalDescriptor> &filter)
{
  // Perform Fourier Transform
  auto inFFT = CreatePayloadInFFT(index, img);
  auto outFFT = CreatePayloadOutFFT(index, inFFT);
  {
    using namespace umpalumpa::fourier_transformation;
    auto &alg = this->GetForwardFFTAlg();
    auto in = AFFT::InputData(inFFT);
    auto out = AFFT::OutputData(outFFT);
    if (!alg.IsInitialized()) {
      auto settings = Settings(Locality::kOutOfPlace, Direction::kForward);
      assert(alg.Init(out, in, settings));
    }
    // std::cout << "Executing FFT on image " << index << "\n";
    assert(alg.Execute(out, in));
  }
  // Perform crop
  auto inCrop = CreatePayloadInCroppedFFT(index, outFFT);
  auto outCrop = CreatePayloadOutCroppedFFT(index, filter.info.GetSize());
  {
    using namespace umpalumpa::fourier_processing;
    using umpalumpa::fourier_transformation::Locality;
    auto &alg = this->GetCropAlg();
    auto in = AFP::InputData(inCrop, filter);
    auto out = AFP::OutputData(outCrop);
    if (!alg.IsInitialized()) {
      auto settings = Settings(Locality::kOutOfPlace);
      settings.SetApplyFilter(true);
      settings.SetNormalize(true);
      assert(alg.Init(out, in, settings));
    }
    // std::cout << "Executing Crop on image " << index << "\n";
    assert(alg.Execute(out, in));
  }
  // NOTE up to here OK for sure, assuming normalization works correctly

  // Release temp data. Input FFT is just reused Payload and we need output Payload for later
  Remove(outFFT.dataInfo);
  return outCrop;
}

template<typename T>
typename FlexAlign<T>::Shift
  FlexAlign<T>::FindMax(size_t i, size_t j, Payload<FourierDescriptor> &correlation)
{
  // perform inverse FFT
  auto outCorrelation = CreatePayloadOutInverseFFT(i, j, correlation);
  {
    using namespace umpalumpa::fourier_transformation;
    auto &alg = this->GetInverseFFTAlg();
    auto in = AFFT::InputData(correlation);
    auto out = AFFT::OutputData(outCorrelation);
    if (!alg.IsInitialized()) {
      auto settings = Settings(Locality::kOutOfPlace, Direction::kInverse);
      assert(alg.Init(out, in, settings));
    }
    // std::cout << "Executing inverse FFT on correlation " << i << " and " << j << "\n";
    assert(alg.Execute(out, in));
  }
  // find maxima
  auto pIn = CreatePayloadMaxIn(i, j, outCorrelation);
  auto empty =
    Payload(LogicalDescriptor(Size(0, 0, 0, 0)), Create(0, DataType::kVoid, true), "Empty");
  auto res = CreatePayloadLocMax(i, j, outCorrelation);
  {
    using namespace umpalumpa::extrema_finder;
    auto &alg = this->GetFindMaxAlg();
    auto in = AExtremaFinder::InputData(pIn);
    auto out = AExtremaFinder::OutputData(empty, res);
    if (!alg.IsInitialized()) {
      // FIXME search around center
      auto settings = Settings(SearchType::kMax, SearchLocation::kEntire, SearchResult::kLocation);
      assert(alg.Init(out, in, settings));
    }
    // std::cout << "Finding maxima in correlation " << i << " and " << j << "\n";
    assert(alg.Execute(out, in));
  }

  Acquire(res);
  // convert index to position
  size_t index = static_cast<size_t>(reinterpret_cast<float *>(res.GetPtr())[0]);
  auto y = index / res.info.GetSize().x;
  auto x = index % res.info.GetSize().x;
  // std::cout << "FindMax correlation " << x << " and " << y << "\n";
  Release(res);
  Remove(res.dataInfo);
  return { x, y };
};

template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::Correlate(size_t i,
  size_t j,
  Payload<FourierDescriptor> &first,
  Payload<FourierDescriptor> &second)
{
  using namespace umpalumpa::correlation;
  // std::cout << "Correlate img " << i << " and " << j << "\n";

  auto pOut = CreatePayloadOutCorrelation(i, j, first);
  auto &alg = this->GetCorrelationAlg();
  auto in = ACorrelation::InputData(first, second);
  auto out = ACorrelation::OutputData(pOut);
  if (!alg.IsInitialized()) {
    auto settings = Settings(CorrelationType::kOneToN);
    assert(alg.Init(out, in, settings));
  }
  assert(alg.Execute(out, in));
  return pOut;
}

template<typename T>
Payload<LogicalDescriptor> FlexAlign<T>::CreatePayloadLocMax(size_t i,
  size_t j,
  const Payload<FourierDescriptor> &correlation)
{
  // std::cout << "Creating Payload for Location of maxima " << i << "-" << j << "\n";
  auto name = "Location of Max " + std::to_string(i) + "-" + std::to_string(j);
  auto type = DataType::kFloat;
  auto size = Size(1, 1, 1, correlation.info.GetSize().n);
  auto ld = LogicalDescriptor(size);
  auto bytes = ld.Elems() * Sizeof(type);
  return Payload(ld, Create(bytes, type, false), name);
}

template<typename T>
Payload<LogicalDescriptor> FlexAlign<T>::CreatePayloadMaxIn(size_t i,
  size_t j,
  const Payload<FourierDescriptor> &correlation)
{
  // std::cout << "Creating Payload for Location of maxima (in) " << i << "-" << j << "\n";
  auto name = "Location of Max (in) " + std::to_string(i) + "-" + std::to_string(j);
  // we 'convert' correlation payload to logical payload
  auto ld = LogicalDescriptor(correlation.info.GetSize(), correlation.info.GetPadding());
  return Payload(ld, correlation.dataInfo.CopyWithPtr(correlation.GetPtr()), name);
}


template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::CreatePayloadOutCorrelation(size_t i,
  size_t j,
  const Payload<FourierDescriptor> &inFFT)
{
  // std::cout << "Creating Payload for correlation (out) " << i << "-" << j << "\n";
  auto name = "Image " + std::to_string(i) + "-" + std::to_string(j);
  // result of the correlation is only intermediary
  return Payload(
    // FIXME this should be temp data
    inFFT.info,
    Create(inFFT.dataInfo.GetBytes(), inFFT.dataInfo.GetType(), false),
    name);
};

template<typename T>
Payload<LogicalDescriptor> FlexAlign<T>::CreatePayloadImage(size_t index, const Size &size)
{
  // std::cout << "Creating Payload for image " << index << "\n";
  auto ld = LogicalDescriptor(size);
  auto type = GetDataType();
  auto bytes = ld.Elems() * Sizeof(type);
  return Payload(ld, Create(bytes, type, false), "Image " + std::to_string(index));
};

template<typename T> Payload<LogicalDescriptor> FlexAlign<T>::CreatePayloadFilter(const Size &size)
{
  // std::cout << "Creating Payload for filter\n";
  auto ld = LogicalDescriptor(size);
  auto type = GetDataType();
  auto bytes = ld.Elems() * Sizeof(type);
  auto payload = Payload(ld, Create(bytes, type, false), "Filter");
  // fill the filter
  auto start = reinterpret_cast<T *>(payload.GetPtr());
  std::fill(start, start + ld.GetSize().total, static_cast<T>(1));
  return payload;
};

template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::CreatePayloadInFFT(size_t index,
  const Payload<LogicalDescriptor> &img)
{
  // std::cout << "Creating Payload for FFT (in) " << index << "\n";
  // we 'convert' image payload to FFT payload
  auto ld = FourierDescriptor(img.info.GetSize(), img.info.GetPadding());
  return Payload(ld, img.dataInfo.CopyWithPtr(img.GetPtr()), "FFT (in) " + std::to_string(index));
};

template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::CreatePayloadOutFFT(size_t index,
  const Payload<FourierDescriptor> &inFFT)
{
  // std::cout << "Creating Payload for FFT (out) " << index << "\n";
  auto ld = FourierDescriptor(inFFT.info.GetSize(),
    inFFT.info.GetPadding(),
    umpalumpa::data::FourierDescriptor::FourierSpaceDescriptor());
  auto type = GetComplexDataType();
  auto bytes = ld.Elems() * Sizeof(type);
  // result of the FFT is only intermediary
  return Payload(ld, Create(bytes, type, true), "FFT (in) " + std::to_string(index));
};

template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::CreatePayloadOutInverseFFT(size_t i,
  size_t j,
  const Payload<FourierDescriptor> &inFFT)
{
  // std::cout << "Creating Payload for FFT (out) " << i << "-" << j << "\n";
  auto ld = FourierDescriptor(inFFT.info.GetSpatialSize(), inFFT.info.GetPadding());
  auto type = GetDataType();
  auto bytes = ld.Elems() * Sizeof(type);
  auto name = "FFT (out) " + std::to_string(i) + "-" + std::to_string(j);
  // result of the FFT is only intermediary
  // FIXME this should be intermediary
  return Payload(ld, Create(bytes, type, false), name);
};

template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::CreatePayloadInCroppedFFT(size_t index,
  const Payload<FourierDescriptor> &inFFT)
{
  // std::cout << "Creating Payload for Crop (in) " << index << "\n";
  // we don't need to change anything
  return Payload(
    inFFT.info, inFFT.dataInfo.CopyWithPtr(inFFT.GetPtr()), "Crop (in) " + std::to_string(index));
}

template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::CreatePayloadOutCroppedFFT(size_t index, const Size &size)
{
  // std::cout << "Creating Payload for Crop (out) " << index << "\n";
  auto ld = FourierDescriptor(size,
    umpalumpa::data::PaddingDescriptor(),
    umpalumpa::data::FourierDescriptor::FourierSpaceDescriptor());
  auto type = GetComplexDataType();
  auto bytes = ld.Elems() * Sizeof(type);
  // result of the crop is for long term storage
  return Payload(ld, Create(bytes, type, false), "Crop (out) " + std::to_string(index));
}

template<typename T>
void FlexAlign<T>::GenerateClockArms(size_t index,
  const Payload<LogicalDescriptor> &p,
  const Size &armSize,
  size_t posX,
  size_t posY)
{
  assert(p.IsValid() && !p.IsEmpty());

  std::cout << "Generated shift of img " << index << " is [" << posX << ", " << posY << "]\n";

  assert(posX + armSize.x < p.info.GetSize().x);
  assert(posY + armSize.y < p.info.GetSize().y);

  auto &imgSize = p.info.GetSize();
  // draw vertical line
  for (size_t y = posY; y < posY + armSize.y; ++y) {
    size_t index = y * imgSize.x + posX;
    reinterpret_cast<T *>(p.GetPtr())[index] = 1;
  }
  // draw horizontal line
  for (size_t x = posX; x < posY + armSize.x; ++x) {
    size_t index = posY * imgSize.x + x;
    reinterpret_cast<T *>(p.GetPtr())[index] = 1;
  }
}

template<typename T> DataType FlexAlign<T>::GetDataType() const
{
  if (std::is_same<T, float>::value) {
    return DataType::kFloat;
  } else if (std::is_same<T, double>::value) {
    return DataType::kDouble;
  }
  return DataType::kVoid;// unsupported
}

template<typename T> DataType FlexAlign<T>::GetComplexDataType() const
{
  if (std::is_same<T, float>::value) {
    return DataType::kComplexFloat;
  } else if (std::is_same<T, double>::value) {
    return DataType::kComplexDouble;
  }
  return DataType::kVoid;// unsupported
}

template class FlexAlign<float>;