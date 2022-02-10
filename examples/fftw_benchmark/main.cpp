#include <libumpalumpa/algorithms/fourier_transformation/fft_cpu.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <vector>
#include <numeric>
#include <chrono>

using namespace umpalumpa;


template<bool isForward> auto run(const data::Size &s, size_t threads)
{
  using namespace data;
  using namespace fourier_transformation;
  using namespace std::chrono;
  auto spacial = [&s]() {
    auto ld = FourierDescriptor(s);
    auto type = DataType::Get<float>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = PhysicalDescriptor(calloc(bytes, 1), bytes, type, ManagedBy::Manually, nullptr);
    return Payload(ld, std::move(pd), "");
  }();
  auto fourier = [&s]() {
    auto ld = FourierDescriptor(s, {}, FourierDescriptor::FourierSpaceDescriptor());
    auto type = DataType::Get<std::complex<float>>();
    auto bytes = ld.Elems() * type.GetSize();
    auto pd = PhysicalDescriptor(calloc(bytes, 1), bytes, type, ManagedBy::Manually, nullptr);
    return Payload(ld, std::move(pd), "");
  }();
  auto alg = fourier_transformation::FFTCPU();
  auto in = AFFT::InputData(isForward ? spacial : fourier);
  auto out = AFFT::OutputData(isForward ? fourier : spacial);
  auto settings =
    Settings(Locality::kOutOfPlace, isForward ? Direction::kForward : Direction::kInverse, threads);
  if (!alg.Init(out, in, settings)) { spdlog::error("Initialization of the FFT algorithm failed"); }
  auto start = high_resolution_clock::now();

  if (!alg.Execute(out, in)) { spdlog::error("Execution of the FFT algorithm failed"); }
  auto duration =
    duration_cast<std::chrono::microseconds>(high_resolution_clock::now() - start).count();

  free(spacial.GetPtr());
  free(fourier.GetPtr());
  return duration;
}

template<bool isForward> void test(const std::vector<data::Size> &sizes)
{
  const size_t reps = 3;
  std::vector<size_t> batches = { 1, 5, 10, 15, 20, 40 };
  std::vector<size_t> threads(24);
  std::iota(threads.begin(), threads.end(), 1);

  for (auto &s : sizes) {
    for (auto b : batches) {
      for (auto t : threads) {
        auto testSize = s.CopyFor(b);
        auto msTotal = 0;
        for (size_t r = 0; r < reps; ++r) { msTotal += run<isForward>(testSize, t); }
        spdlog::info(
          "{}, size {} x {} batch {}, threads: {}, repetitions: {}, total time (us): {}, avg time "
          "(us): {}",
          isForward ? "Forward FT" : "Inverse FT",
          testSize.x,
          testSize.y,
          testSize.n,
          t,
          reps,
          msTotal,
          msTotal / reps);
      }
    }
  }
}

int main()
{

  std::vector<data::Size> fft;
  fft.emplace_back(4096, 4096, 1, 40);
  fft.emplace_back(3584, 3584, 1, 40);
  fft.emplace_back(7168, 7168, 1, 40);
  fft.emplace_back(5600, 4000, 1, 30);
  fft.emplace_back(11520, 7776, 1, 20);
  test<true>(fft);

  std::vector<data::Size> ift;
  ift.emplace_back(972, 972, 1, 736);
  ift.emplace_back(864, 864, 1, 780);
  ift.emplace_back(1728, 1728, 1, 233);
  ift.emplace_back(1296, 972, 1, 435);
  ift.emplace_back(2646, 1792, 1, 146);
  test<false>(ift);

  return 0;
}