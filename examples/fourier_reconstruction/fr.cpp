#include "fr.hpp"
#include <cassert>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <libumpalumpa/utils/payload.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space_generator.hpp>
#include <iostream>

template<typename T>
void FourierReconstruction<T>::Execute(const umpalumpa::data::Size &imgSize,
  size_t noOfSymmetries,
  size_t batchSize)
{
  assert(imgSize.x % 2 == 0);// we can process only odd size of the images

  auto imgBatchSize = umpalumpa::data::Size(imgSize.x, imgSize.y, 1, batchSize);
  auto volumeSize = umpalumpa::data::Size(imgSize.x + 1, imgSize.y + 1, imgSize.y + 1, 1);
  auto imgCroppedBatchSize = umpalumpa::data::Size(
    imgSize.x / 2, imgSize.y, 1, batchSize);// This should probably be .x / 2 + 1 (i.e. normal FFT
                                            // size), but in Xmipp it's like that
  auto traverseSpaceBatchSize = umpalumpa::data::Size(1, 1, 1, batchSize * noOfSymmetries);

  spdlog::info(
    "\nRunning Fourier Reconstruction.\nImage size: {}*{} ({})\nBatch: {}\nSymmetries: {}",
    imgSize.x,
    imgSize.y,
    imgSize.n,
    batchSize,
    noOfSymmetries);

  auto symmetries = GenerateSymmetries(noOfSymmetries);
  auto filter = CreatePayloadFilter(imgCroppedBatchSize);
  auto settings = umpalumpa::fourier_reconstruction::Settings{};
  settings.SetType(umpalumpa::fourier_reconstruction::Settings::Type::kPrecise);
  settings.SetInterpolation(umpalumpa::fourier_reconstruction::Settings::Interpolation::kDynamic);
  auto volume = CreatePayloadVolume(volumeSize);
  auto weight = CreatePayloadWeight(volumeSize);
  // FIXME init volume and weight to 0
  auto table = CreatePayloadBlobTable(settings);

  for (size_t i = 0; i < imgSize.n; i += batchSize) {
    auto name = std::to_string(i) + "-" + std::to_string(i + batchSize - 1);
    spdlog::info("Loop {}", name);
    auto img = CreatePayloadImage(imgBatchSize, name);
    GenerateData(i, img);
    auto space = CreatePayloadTraverseSpace(traverseSpaceBatchSize, name);
    GenerateTraverseSpaces(imgCroppedBatchSize, volumeSize, space, symmetries, settings);
    auto fft = ConvertToFFT(img, name);
    auto croppedFFT = Crop(fft, filter, name);
    InsertToVolume(croppedFFT, volume, weight, space, table, settings);
    RemovePD(img.dataInfo, true);
    RemovePD(fft.dataInfo, false);
    RemovePD(space.dataInfo, true);
    RemovePD(croppedFFT.dataInfo, false);
  }

  GetFRAlg().Synchronize(); // wait till the work is done
  
  // Show results
  // Print(volume, "Volume data");
  // Print(weight, "Weight data");

  RemovePD(table.dataInfo, true);
  RemovePD(weight.dataInfo, true);
  RemovePD(volume.dataInfo, true);
  RemovePD(filter.dataInfo, false);
}

template<typename T>
template<typename U>
void FourierReconstruction<T>::Print(const Payload<U> &p, const std::string &name) {
  Acquire(p.dataInfo);
  std::cout << name << "\n";
  umpalumpa::utils::PrintData(std::cout , p);
  std::cout << "\n";
  Release(p.dataInfo);
}

template<typename T> auto FourierReconstruction<T>::GenerateSymmetries(size_t count)
{

  std::vector<Matrix3x3> res;
  res.reserve(count);
  for (size_t i = 0; i < count; ++i) { res.emplace_back(GenerateMatrix()); }
  spdlog::info("{} symmetries generated", res.size());
  return res;
}

template<typename T> auto FourierReconstruction<T>::CreatePayloadVolume(const Size &size)
{
  auto fd = FourierDescriptor::FourierSpaceDescriptor{};
  fd.hasSymetry = true;
  auto ld = FourierDescriptor(size, umpalumpa::data::PaddingDescriptor(), fd);
  auto type = DataType::Get<std::complex<T>>();
  auto bytes = ld.Elems() * type.GetSize();
  return umpalumpa::data::Payload(ld, CreatePD(bytes, type, true, true), "Volume in FD");
}

template<typename T> auto FourierReconstruction<T>::CreatePayloadWeight(const Size &size)
{
  auto ld = LogicalDescriptor(size);
  auto type = DataType::Get<T>();
  auto bytes = ld.Elems() * type.GetSize();
  return umpalumpa::data::Payload(ld, CreatePD(bytes, type, true, true), "Weight in FD");
}

template<typename T>
auto FourierReconstruction<T>::CreatePayloadBlobTable(
  const umpalumpa::fourier_reconstruction::Settings &settings)
{
  using umpalumpa::fourier_reconstruction::Settings;
  auto count = settings.GetInterpolation() == Settings::Interpolation::kLookup ? 10000 : 0;
  auto ld = LogicalDescriptor(Size(count, 1, 1, 1));
  auto type = DataType::Get<T>();
  auto bytes = ld.Elems() * type.GetSize();
  return umpalumpa::data::Payload(ld, CreatePD(bytes, type, true, true), "Interpolation table");
}

template<typename T> size_t FourierReconstruction<T>::GetAvailableCores() const
{
  return std::thread::hardware_concurrency() / 2;// assuming HT is available
}

template<typename T>
Payload<FourierDescriptor> FourierReconstruction<T>::ConvertToFFT(
  const Payload<LogicalDescriptor> &img,
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
  auto &alg = this->GetFFTAlg();
  auto in = AFFT::InputData(inFFT);
  auto out = AFFT::OutputData(outFFT);
  if (!alg.IsInitialized()) {
    auto settings =
      Settings(Locality::kOutOfPlace, Direction::kForward, std::min(1ul, GetAvailableCores()));
    if (!alg.Init(out, in, settings)) {
      spdlog::error("Initialization of the FFT algorithm failed");
    }
  }
  if (!alg.Execute(out, in)) { spdlog::error("Execution of the FFT algorithm failed"); }
  return outFFT;
}

template<typename T>
Payload<FourierDescriptor> FourierReconstruction<T>::Crop(const Payload<FourierDescriptor> &fft,
  Payload<LogicalDescriptor> &filter,
  const std::string &name)
{
  auto inCrop = [&fft, &name]() {
    return Payload(fft.info, fft.dataInfo.CopyWithPtr(fft.GetPtr()), "Crop (in) " + name);
  }();
  auto outCrop = [&filter, &fft, &name, this]() {
    const auto &imgSize = fft.info.GetSpatialSize();
    // croppedSize is selected such as its FFT is x / 2 of the original size. This might be a bug
    // in Xmipp
    auto croppedSize = umpalumpa::data::Size(imgSize.x - 2, imgSize.y, 1, imgSize.n);
    auto ld = FourierDescriptor(croppedSize,
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
    settings.SetApplyFilter(false);
    settings.SetCenter(true);
    settings.SetNormalize(true);
    settings.SetShift(true);
    if (!alg.Init(out, in, settings)) {
      spdlog::error("Initialization of the Crop algorithm failed");
    }
  }
  if (!alg.Execute(out, in)) { spdlog::error("Execution of the Crop algorithm failed"); }
  return outCrop;
}

template<typename T>
void FourierReconstruction<T>::InsertToVolume(Payload<FourierDescriptor> &fft,
  Payload<FourierDescriptor> &volume,
  Payload<LogicalDescriptor> &weight,
  Payload<LogicalDescriptor> &traverseSpace,
  Payload<LogicalDescriptor> &table,
  const umpalumpa::fourier_reconstruction::Settings &settings)
{
  using namespace umpalumpa::fourier_reconstruction;
  auto &alg = this->GetFRAlg();
  auto in = AFR::InputData(fft, volume, weight, traverseSpace, table);
  auto out = AFR::OutputData(volume, weight);
  if (!alg.IsInitialized()) {
    if (!alg.Init(out, in, settings)) {
      spdlog::error("Initialization of the Fourier Reconstruction algorithm failed");
    }
    if (settings.GetInterpolation() == Settings::Interpolation::kLookup) {
      AFR::FillBlobTable(in, settings);
    }
  }
  if (!alg.Execute(out, in)) {
    spdlog::error("Execution of the Fourier Reconstruction algorithm failed");
  }
}

template<typename T>
Payload<LogicalDescriptor> FourierReconstruction<T>::CreatePayloadImage(const Size &size,
  const std::string &name)
{
  auto ld = LogicalDescriptor(size);
  auto type = DataType::Get<T>();
  auto bytes = ld.Elems() * type.GetSize();
  return Payload(ld, CreatePD(bytes, type, true, true), "Image(s) " + name);
};

template<typename T>
Payload<LogicalDescriptor> FourierReconstruction<T>::CreatePayloadTraverseSpace(const Size &size,
  const std::string &name)
{
  using umpalumpa::fourier_reconstruction::TraverseSpace;
  auto ld = LogicalDescriptor(size);
  auto type = DataType::Get<TraverseSpace>();
  auto bytes = ld.Elems() * type.GetSize();
  return Payload(ld, CreatePD(bytes, type, true, true), "Traverse space(s) " + name);
};

template<typename T>
Payload<LogicalDescriptor> FourierReconstruction<T>::CreatePayloadFilter(const Size &size)
{
  auto ld = LogicalDescriptor(size.CopyFor(1));
  auto type = DataType::Get<void>();
  auto bytes = ld.Elems() * type.GetSize();
  auto payload = Payload(ld, CreatePD(0, type, false, false), "Filter");
  return payload;
};

template<typename T> void FillRandom(T *dst, size_t bytes)
{
  std::iota(dst, dst + bytes / sizeof(T), T(1));
}

template<typename T>
void FourierReconstruction<T>::GenerateData(size_t, const Payload<LogicalDescriptor> &p)
{
  assert(p.IsValid() && !p.IsEmpty());
  Acquire(p.dataInfo);
  auto *ptr = reinterpret_cast<T *>(p.GetPtr());
  FillRandom(ptr, p.dataInfo.GetBytes());
  // std::fill(ptr, ptr + p.info.GetSize().total, static_cast<T>(1));
  spdlog::info("Image data generated");
  Release(p.dataInfo);
}

template<typename T>
void FourierReconstruction<T>::GenerateTraverseSpaces(
  const umpalumpa::data::Size &imgCroppedBatchSize,
  const umpalumpa::data::Size &volumeSize,
  const Payload<LogicalDescriptor> &p,
  const std::vector<Matrix3x3> &symmetries,
  const umpalumpa::fourier_reconstruction::Settings &settings)
{
  assert(p.IsValid() && !p.IsEmpty());
  assert(p.info.Elems() == imgCroppedBatchSize.n * symmetries.size());
  using umpalumpa::fourier_reconstruction::TraverseSpace;
  using umpalumpa::fourier_reconstruction::Settings;

  Acquire(p.dataInfo);
  size_t counter = 0;
  for (size_t img = 0; img < imgCroppedBatchSize.n; ++img) {
    auto m = GenerateMatrix();
    for (const auto &s : symmetries) {
      auto transform = Multiply(m, s);
      auto &ptr = reinterpret_cast<TraverseSpace *>(p.GetPtr())[counter++];
      umpalumpa::fourier_reconstruction::ComputeTraverseSpace(imgCroppedBatchSize.x,
        imgCroppedBatchSize.y,
        reinterpret_cast<float(*)[3]>(transform.data()),
        ptr,
        volumeSize.x - 1,
        volumeSize.y - 1,
        settings.GetType() == Settings::Type::kFast,
        settings.GetBlobRadius(),
        1.f);
      ptr.projectionIndex = img;
    }
  }
  Release(p.dataInfo);
  spdlog::info("Traverse spaces generated");
}

template class FourierReconstruction<float>;