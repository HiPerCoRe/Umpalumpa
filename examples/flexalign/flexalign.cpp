#include "flexalign.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <type_traits>
#include <libumpalumpa/utils/payload.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

template<typename T> void FlexAlign<T>::Execute(const umpalumpa::data::Size &sizeAll)
{
  assert(sizeAll.x > 5);
  assert(sizeAll.y > 5);
  assert(sizeAll.z == 1);

  auto sizeSingle = sizeAll.CopyFor(1);
  auto sizeSingleCrop = Size(sizeSingle.x / 2, sizeSingle.y / 2, sizeSingle.z, 1);
  auto sizeCross = Size(3, 3, 1, 1);
  auto scaleX = static_cast<float>(sizeSingle.x) / static_cast<float>(sizeSingleCrop.x);
  auto scaleY = static_cast<float>(sizeSingle.y) / static_cast<float>(sizeSingleCrop.y);

  auto filter = CreatePayloadFilter(sizeSingleCrop);

  auto imgs = std::vector<Payload<LogicalDescriptor>>();
  imgs.reserve(sizeAll.n);

  auto ffts = std::vector<Payload<FourierDescriptor>>();
  ffts.reserve(sizeAll.n);

  for (size_t j = 0; j < sizeAll.n; ++j) {
    auto name = std::to_string(j);
    auto &img = imgs.emplace_back(CreatePayloadImage(sizeSingle, name));
    GenerateClockArms(j, img, sizeCross, j, j);
    auto fft = ConvertToFFT(img, name);
    ffts.emplace_back(Crop(fft, filter, name));
    RemovePD(fft.dataInfo);
    for (size_t i = 0; i < j; ++i) {
      auto correlation = Correlate(i, j, ffts.at(i), ffts.at(j));
      auto shift = FindMax(i, j, correlation);
      Remove(correlation.dataInfo);
      // reported shift is position in the 2D image, where center of that image
      // has position [0, 0];
      // To get the right shift, we need to shift by half of the cropped image
      // Since we cropped the image in the Fourier domain and performed IFFT, we performed
      // downscaling To get the rigth shift, we have to adjust the scale.
      auto normShift = Transform(shift, scaleX, scaleY, sizeSingleCrop.x / 2, sizeSingleCrop.y / 2);
      LogResult(i, j, normShift);
    }
  }
  Synchronize();
  // Release allocated data. Payloads themselves don't need any extra handling
  for (const auto &p : ffts) { RemovePD(p.dataInfo); }
  for (const auto &p : imgs) { RemovePD(p.dataInfo); }
  RemovePD(filter.dataInfo);
}

template<typename T> void FlexAlign<T>::LogResult(size_t i, size_t j, const Shift &shift)
{
  const auto expectedShift = static_cast<float>(j - i);
  const auto maxDelta =
    std::max(std::abs(shift.x - expectedShift), std::abs(shift.y - expectedShift));
  const auto level = [maxDelta]() {
    constexpr auto delta1 = 0.1f;
    constexpr auto delta2 = 0.5f;
    if (maxDelta < delta1) return spdlog::level::info;
    if (maxDelta < delta2) return spdlog::level::warn;
    return spdlog::level::err;
  }();
  spdlog::log(level,
    "Shift of img {} and {} is [{}, {}] (expected [{}, {}])",
    i,
    j,
    shift.x,
    shift.y,
    expectedShift,
    expectedShift);
}

template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::ConvertToFFT(const Payload<LogicalDescriptor> &img,
  const std::string &name)
{
  // Perform Fourier Transform
  auto inFFT = [&img, &name]() {
    auto ld = FourierDescriptor(img.info.GetSize(), img.info.GetPadding());
    return Payload(ld, img.dataInfo.CopyWithPtr(img.GetPtr()), "FFT (in) " + name);
  }();
  auto outFFT = [&inFFT, &name, this]() {
    auto ld = FourierDescriptor(inFFT.info.GetSize(),
      inFFT.info.GetPadding(),
      umpalumpa::data::FourierDescriptor::FourierSpaceDescriptor());
    auto type = GetComplexDataType();
    auto bytes = ld.Elems() * Sizeof(type);
    // result of the FFT is only intermediary
    return Payload(ld, CreatePD(bytes, type, false), "FFT (out) " + name);
  }();
  using namespace umpalumpa::fourier_transformation;
  auto &alg = this->GetForwardFFTAlg();
  auto in = AFFT::InputData(inFFT);
  auto out = AFFT::OutputData(outFFT);
  if (!alg.IsInitialized()) {
    auto settings = Settings(Locality::kOutOfPlace, Direction::kForward);
    if (!alg.Init(out, in, settings)) {
      spdlog::error("Initialization of the FFT algorithm failed");
    }
  }
  if (!alg.Execute(out, in)) { spdlog::error("Execution of the FFT algorithm failed"); }
  return outFFT;
}


template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::Crop(const Payload<FourierDescriptor> &fft,
  Payload<LogicalDescriptor> &filter,
  const std::string &name)
{
  auto inCrop = [&fft, &name]() {
    return Payload(fft.info, fft.dataInfo.CopyWithPtr(fft.GetPtr()), "Crop (in) " + name);
  }();
  auto outCrop = [&filter, &name, this]() {
    auto ld = FourierDescriptor(filter.info.GetSize(),
      umpalumpa::data::PaddingDescriptor(),
      umpalumpa::data::FourierDescriptor::FourierSpaceDescriptor());
    auto type = GetComplexDataType();
    auto bytes = ld.Elems() * Sizeof(type);
    // result of the crop is for long term storage
    return Payload(ld, Create(bytes, type, false), "Crop (out) " + name);
  }();
  using namespace umpalumpa::fourier_processing;
  using umpalumpa::fourier_transformation::Locality;
  auto &alg = this->GetCropAlg();
  auto in = AFP::InputData(inCrop, filter);
  auto out = AFP::OutputData(outCrop);
  if (!alg.IsInitialized()) {
    auto settings = Settings(Locality::kOutOfPlace);
    settings.SetApplyFilter(true);
    settings.SetNormalize(true);
    if (!alg.Init(out, in, settings)) {
      spdlog::error("Initialization of the Crop algorithm failed");
    }
  }
  // std::cout << "Executing Crop on image " << index << "\n";
  if (!alg.Execute(out, in)) { spdlog::error("Execution of the Crop algorithm failed"); }
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
      if (!alg.Init(out, in, settings)) {
        spdlog::error("Initialization of the FFT algorithm failed");
      }
    }
    // std::cout << "Executing inverse FFT on correlation " << i << " and " << j << "\n";
    if (!alg.Execute(out, in)) { spdlog::error("Execution of the FFT algorithm failed"); }
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
      auto settings =
        Settings(ExtremaType::kMax, Location::kEntire, Result::kLocation, Precision::k3x3);
      if (!alg.Init(out, in, settings)) {
        spdlog::error("Initialization of the Extrema Finder algorithm failed");
      }
    }
    // std::cout << "Finding maxima in correlation " << i << " and " << j << "\n";
    if (!alg.Execute(out, in)) {
      spdlog::error("Execution of the Extrema Finder algorithm failed");
    }
  }

  Acquire(res.dataInfo);
  auto x = reinterpret_cast<float *>(res.GetPtr())[0];
  auto y = reinterpret_cast<float *>(res.GetPtr())[1];
  // std::cout << "FindMax correlation " << x << " and " << y << "\n";
  Release(res.dataInfo);
  Remove(res.dataInfo);
  Remove(outCorrelation.dataInfo);
  Remove(empty.dataInfo);
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
    if (!alg.Init(out, in, settings)) {
      spdlog::error("Initialization of the Correlation algorithm failed");
    }
  }
  if (!alg.Execute(out, in)) { spdlog::error("Execution of the Correlation algorithm failed"); }
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
  auto size = Size(2, 1, 1, correlation.info.GetSize().n);
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
Payload<LogicalDescriptor> FlexAlign<T>::CreatePayloadImage(const Size &size,
  const std::string &name)
{
  auto ld = LogicalDescriptor(size);
  auto type = GetDataType();
  auto bytes = ld.Elems() * Sizeof(type);
  return Payload(ld, CreatePD(bytes, type, true), "Image " + name);
};

template<typename T> Payload<LogicalDescriptor> FlexAlign<T>::CreatePayloadFilter(const Size &size)
{
  auto ld = LogicalDescriptor(size);
  auto type = GetDataType();
  auto bytes = ld.Elems() * Sizeof(type);
  auto payload = Payload(ld, CreatePD(bytes, type, true), "Filter");
  // fill the filter
  Acquire(payload.dataInfo);
  auto start = reinterpret_cast<T *>(payload.GetPtr());
  std::fill(start, start + ld.GetSize().total, static_cast<T>(1));
  Release(payload.dataInfo);
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
  return Payload(ld, CreatePD(bytes, type, false), "FFT (in) " + std::to_string(index));
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
  Acquire(p.dataInfo);
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
  Release(p.dataInfo);
}

template<typename T> constexpr DataType FlexAlign<T>::GetDataType() const
{
  if (std::is_same<T, float>::value) {
    return DataType::kFloat;
  } else if (std::is_same<T, double>::value) {
    return DataType::kDouble;
  }
  return DataType::kVoid;// unsupported
}

template<typename T> constexpr DataType FlexAlign<T>::GetComplexDataType() const
{
  if (std::is_same<T, float>::value) {
    return DataType::kComplexFloat;
  } else if (std::is_same<T, double>::value) {
    return DataType::kComplexDouble;
  }
  return DataType::kVoid;// unsupported
}

template class FlexAlign<float>;