#include "flexalign.hpp"
#include <cassert>
#include <libumpalumpa/system_includes/spdlog.hpp>

template<typename T> void FlexAlign<T>::Execute(const umpalumpa::data::Size &sizeAll)
{
  assert(sizeAll.x > 5);
  assert(sizeAll.y > 5);
  assert(sizeAll.z == 1);

  const size_t batch = 5;
  assert(0 == sizeAll.n % batch);

  auto sizeBatch = sizeAll.CopyFor(batch);
  auto sizeBatchCrop = Size(sizeBatch.x / 2, sizeBatch.y / 2, sizeBatch.z, sizeBatch.n);
  // auto sizeBatchCrop = Size(928, 928, 1, 1);
  auto sizeCross = Size(3, 3, 1, 1);
  auto scaleX = static_cast<float>(sizeBatch.x) / static_cast<float>(sizeBatchCrop.x);
  auto scaleY = static_cast<float>(sizeBatch.y) / static_cast<float>(sizeBatchCrop.y);

  auto filter = CreatePayloadFilter(sizeBatchCrop);

  auto imgs = std::vector<Payload<LogicalDescriptor>>();
  imgs.reserve(sizeAll.n);

  auto ffts = std::vector<Payload<FourierDescriptor>>();
  ffts.reserve(sizeAll.n);

  auto shifts = std::vector<Payload<LogicalDescriptor>>();
  shifts.reserve(NoOfBatches(sizeAll, batch));

  // Preallocate memory. This can be quite expensive, because e.g. cudaHostAlloc is
  // synchronizing, i.e. it can slow down the execution later on
  for (size_t j = 0; j < sizeAll.n; j += batch) {
    auto name = std::to_string(j) + "-" + std::to_string(j + batch - 1);
    auto &img = imgs.emplace_back(CreatePayloadImage(sizeBatch, name));
  }

  for (size_t j = 0; j < sizeAll.n; j += batch) {
    auto name = std::to_string(j) + "-" + std::to_string(j + batch - 1);
    auto &img = imgs.at(j / batch);
    GenerateClockArms(j, img, sizeCross, j, j);
    auto fft = ConvertToFFT(img, name);
    ffts.emplace_back(Crop(fft, filter, name));
    RemovePD(fft.dataInfo, false);
    for (size_t i = 0; i <= j; i += batch) {
      auto name = std::to_string(i) + "-" + std::to_string(i + batch - 1) + "<->"
                  + std::to_string(j) + "-" + std::to_string(j + batch - 1);
      auto correlation = Correlate(ffts.at(i / batch), ffts.at(j / batch), name);
      auto ifft = ConvertFromFFT(correlation, name);
      RemovePD(correlation.dataInfo, false);
      shifts.emplace_back(FindMax(ifft, name));
      RemovePD(ifft.dataInfo, false);
    }
  }
  // wait for results and process them
  assert(shifts.size() == NoOfBatches(sizeAll, batch));
  for (size_t j = 0, counter = 0; j < sizeAll.n; j += batch) {
    for (size_t i = 0; i <= j; i += batch, ++counter) {
      auto shift = ExtractShift(shifts.at(counter));
      // reported shift is position in the 2D image, where center of that image
      // has position [0, 0];
      // To get the right shift, we need to shift by half of the cropped image
      // Since we cropped the image in the Fourier domain and performed IFFT, we performed
      // downscaling To get the rigth shift, we have to adjust the scale.
      auto normShift = Transform(shift, scaleX, scaleY, sizeBatchCrop.x / 2, sizeBatchCrop.y / 2);
      LogResult(i, j, batch, normShift);
    }
  }
  // Release allocated data. Payloads themselves don't need any extra handling
  for (const auto &p : ffts) { RemovePD(p.dataInfo, false); }
  for (const auto &p : imgs) { RemovePD(p.dataInfo, true); }

  RemovePD(filter.dataInfo, true);
}

template<typename T> size_t FlexAlign<T>::GetAvailableCores() const
{
  return std::thread::hardware_concurrency() / 2;// assuming HT is available
}

template<typename T>
std::vector<typename FlexAlign<T>::Shift> FlexAlign<T>::ExtractShift(
  const Payload<LogicalDescriptor> &shift)
{
  std::vector<Shift> res;
  Acquire(shift.dataInfo);
  for (size_t n = 0; n < shift.info.GetSize().n; ++n) {
    auto x = reinterpret_cast<float *>(shift.GetPtr())[2 * n];
    auto y = reinterpret_cast<float *>(shift.GetPtr())[2 * n + 1];
    res.push_back({ x, y });
  }
  Release(shift.dataInfo);
  RemovePD(shift.dataInfo, false);
  return res;
}

template<typename T>
void FlexAlign<T>::LogResult(size_t i, size_t j, size_t batch, const std::vector<Shift> &shift)
{
  assert(NoOfCorrelations(batch, i == j) == shift.size());
  size_t counter = 0;
  for (size_t idxI = i; idxI < (i + batch); ++idxI) {
    for (size_t idxJ = j; idxJ < (j + batch); ++idxJ) {
      if (idxI >= idxJ) continue;
      const auto expectedShift = static_cast<float>(idxJ - idxI);
      auto actualShift = shift.at(counter);
      const auto maxDelta =
        std::max(std::abs(actualShift.x - expectedShift), std::abs(actualShift.y - expectedShift));
      const auto level = [maxDelta]() {
        constexpr auto delta1 = 0.1f;
        constexpr auto delta2 = 0.5f;
        if (maxDelta < delta1) return spdlog::level::info;
        if (maxDelta < delta2) return spdlog::level::warn;
        return spdlog::level::err;
      }();
      spdlog::log(level,
        "Shift of img {} and {} is [{}, {}] (expected [{}, {}])",
        idxI,
        idxJ,
        actualShift.x,
        actualShift.y,
        expectedShift,
        expectedShift);
      counter++;
    }
  }
}

template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::ConvertToFFT(const Payload<LogicalDescriptor> &img,
  const std::string &name)
{
  auto inFFT = [&img, &name]() {
    auto ld = FourierDescriptor(img.info.GetSize(), img.info.GetPadding());
    return Payload(ld, img.dataInfo.CopyWithPtr(img.GetPtr()), "FFT (in) " + name);
  }();
  auto outFFT = [&inFFT, &name, this]() {
    auto ld = FourierDescriptor(inFFT.info.GetSize(),
      inFFT.info.GetPadding(),
      umpalumpa::data::FourierDescriptor::FourierSpaceDescriptor());
    auto type = DataType::Get<std::complex<T>>();
    auto bytes = ld.Elems() * type.GetSize();
    return Payload(ld, CreatePD(bytes, type, false, false), "FFT (out) " + name);
  }();
  using namespace umpalumpa::fourier_transformation;
  auto &alg = this->GetForwardFFTAlg();
  auto in = AFFT::InputData(inFFT);
  auto out = AFFT::OutputData(outFFT);
  if (!alg.IsInitialized()) {
    auto settings =
      Settings(Locality::kOutOfPlace, Direction::kForward, std::min(8ul, GetAvailableCores()));
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
  auto outCrop = [&filter, &fft, &name, this]() {
    auto ld = FourierDescriptor(filter.info.GetSize().CopyFor(fft.info.GetSize().n),
      umpalumpa::data::PaddingDescriptor(),
      umpalumpa::data::FourierDescriptor::FourierSpaceDescriptor());
    auto type = DataType::Get<std::complex<T>>();
    auto bytes = ld.Elems() * type.GetSize();
    return Payload(ld, CreatePD(bytes, type, false, false), "Crop (out) " + name);
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
  if (!alg.Execute(out, in)) { spdlog::error("Execution of the Crop algorithm failed"); }
  return outCrop;
}

template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::ConvertFromFFT(Payload<FourierDescriptor> &correlation,
  const std::string &name)
{
  auto pOut = [&correlation, &name, this]() {
    auto ld = FourierDescriptor(correlation.info.GetSpatialSize(), correlation.info.GetPadding());
    auto type = DataType::Get<T>();
    auto bytes = ld.Elems() * type.GetSize();
    return Payload(ld, CreatePD(bytes, type, false, false), "IFFT (out) " + name);
  }();
  using namespace umpalumpa::fourier_transformation;
  auto &alg = this->GetInverseFFTAlg();
  auto in = AFFT::InputData(correlation);
  auto out = AFFT::OutputData(pOut);
  if (!alg.IsInitialized()) {
    auto settings =
      Settings(Locality::kOutOfPlace, Direction::kInverse, std::min(4ul, GetAvailableCores()));
    if (!alg.Init(out, in, settings)) {
      spdlog::error("Initialization of the IFFT algorithm failed");
    }
  }
  if (!alg.Execute(out, in)) { spdlog::error("Execution of the IFFT algorithm failed"); }
  return pOut;
}

template<typename T>
Payload<LogicalDescriptor> FlexAlign<T>::FindMax(Payload<FourierDescriptor> &outCorrelation,
  const std::string &name)
{
  auto pIn = [&outCorrelation, &name]() {
    auto ld = LogicalDescriptor(outCorrelation.info.GetSize(), outCorrelation.info.GetPadding());
    return Payload(ld,
      outCorrelation.dataInfo.CopyWithPtr(outCorrelation.GetPtr()),
      "Location of Max (in) " + name);
  }();
  auto empty = Payload(
    LogicalDescriptor(Size(0, 0, 0, 0)), CreatePD(0, DataType::Get<void>(), false, false), "Empty");
  auto pOut = [&outCorrelation, &name, this]() {
    auto type = DataType::Get<float>();
    auto size = Size(2, 1, 1, outCorrelation.info.GetSize().n);
    auto ld = LogicalDescriptor(size);
    auto bytes = ld.Elems() * type.GetSize();
    return Payload(ld, CreatePD(bytes, type, true, false), "Location of Max " + name);
  }();
  using namespace umpalumpa::extrema_finder;
  auto &alg = this->GetFindMaxAlg();
  auto in = AExtremaFinder::InputData(pIn);
  auto out = AExtremaFinder::OutputData(empty, pOut);
  if (!alg.IsInitialized()) {
    // FIXME search around center
    auto settings =
      Settings(ExtremaType::kMax, Location::kEntire, Result::kLocation, Precision::k3x3);
    if (!alg.Init(out, in, settings)) {
      spdlog::error("Initialization of the Extrema Finder algorithm failed");
    }
  }
  if (!alg.Execute(out, in)) { spdlog::error("Execution of the Extrema Finder algorithm failed"); }
  RemovePD(empty.dataInfo, false);
  return pOut;
};

template<typename T>
Payload<FourierDescriptor> FlexAlign<T>::Correlate(Payload<FourierDescriptor> &first,
  Payload<FourierDescriptor> &second,
  const std::string &name)
{
  using namespace umpalumpa::correlation;
  auto pOut = [&first, &second, &name, this]() {
    auto &n1 = first.info.GetSize().n;
    auto &n2 = second.info.GetSize().n;
    assert(n1 == n2);
    auto nOut = NoOfCorrelations(n1, &first == &second);
    auto sizeOut = first.info.GetSpatialSize().CopyFor(nOut);
    auto fd = first.info.GetFourierSpaceDescriptor();
    auto ld = FourierDescriptor(sizeOut, umpalumpa::data::PaddingDescriptor(), fd.value());
    auto bytes = ld.Elems() * first.dataInfo.GetType().GetSize();
    auto pd = CreatePD(bytes, first.dataInfo.GetType(), false, false);
    return Payload(ld, std::move(pd), "Correlation of " + name);
  }();
  auto &alg = this->GetCorrelationAlg();
  auto in = ACorrelation::InputData(first, second);
  auto out = ACorrelation::OutputData(pOut);
  if (!alg.IsInitialized()) {
    auto settings = Settings(CorrelationType::kMToN);
    if (!alg.Init(out, in, settings)) {
      spdlog::error("Initialization of the Correlation algorithm failed");
    }
  }
  if (!alg.Execute(out, in)) { spdlog::error("Execution of the Correlation algorithm failed"); }
  return pOut;
}

template<typename T>
Payload<LogicalDescriptor> FlexAlign<T>::CreatePayloadImage(const Size &size,
  const std::string &name)
{
  auto ld = LogicalDescriptor(size);
  auto type = DataType::Get<T>();
  auto bytes = ld.Elems() * type.GetSize();
  return Payload(ld, CreatePD(bytes, type, true, true), "Image(s) " + name);
};

template<typename T> Payload<LogicalDescriptor> FlexAlign<T>::CreatePayloadFilter(const Size &size)
{
  auto ld = LogicalDescriptor(size.CopyFor(1));
  auto type = DataType::Get<T>();
  auto bytes = ld.Elems() * type.GetSize();
  auto payload = Payload(ld, CreatePD(bytes, type, true, true), "Filter");
  // fill the filter
  Acquire(payload.dataInfo);
  auto start = reinterpret_cast<T *>(payload.GetPtr());
  std::fill(start, start + ld.GetSize().total, static_cast<T>(1));
  Release(payload.dataInfo);
  return payload;
};

template<typename T>
void FlexAlign<T>::GenerateClockArms(size_t index,
  const Payload<LogicalDescriptor> &p,
  const Size &armSize,
  size_t startX,
  size_t startY)
{
  assert(p.IsValid() && !p.IsEmpty());

  Acquire(p.dataInfo);
  for (size_t n = 0; n < p.info.GetSize().n; ++n) {
    auto posX = startX + n;
    auto posY = startY + n;
    spdlog::info("Generated shift of img {} is [{}, {}]", index + n, posX, posY);

    assert(posX + armSize.x < p.info.GetSize().x);
    assert(posY + armSize.y < p.info.GetSize().y);

    auto imgSize = p.info.GetSize().CopyFor(1);
    auto offset = imgSize.single * n;

    // draw vertical line
    for (size_t y = posY; y < posY + armSize.y; ++y) {
      size_t i = y * imgSize.x + posX;
      reinterpret_cast<T *>(p.GetPtr())[offset + i] = 1;
    }
    // draw horizontal line
    for (size_t x = posX; x < posY + armSize.x; ++x) {
      size_t i = posY * imgSize.x + x;
      reinterpret_cast<T *>(p.GetPtr())[offset + i] = 1;
    }
  }
  Release(p.dataInfo);
}

template class FlexAlign<float>;