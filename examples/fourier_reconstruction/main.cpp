#include "fr_cpu.hpp"
#include "fr_cuda.hpp"
#ifdef USE_STARPU
#include "fr_starpu.hpp"
#endif
#include <iostream>

int main(int argc, char **argv)
{
  using umpalumpa::fourier_reconstruction::Settings;
  size_t x = 128;
  size_t n = 1000;
  size_t batch = 20;
  size_t symmetries = 5;
  auto type = Settings::Type::kPrecise;
  auto interpolation = Settings::Interpolation::kDynamic;
  if (argc == 7) {
    x = atoi(argv[1]);
    n = atoi(argv[2]);
    batch = atoi(argv[3]);
    symmetries = atoi(argv[4]);
    type = (0 == std::string{ "fast" }.compare(argv[5])) ? Settings::Type::kFast
                                                         : Settings::Type::kPrecise;
    interpolation = (0 == std::string{ "table" }.compare(argv[6]))
                      ? Settings::Interpolation::kLookup
                      : Settings::Interpolation::kDynamic;
  } else {
    std::cout
      << "Using default values.\n\n"
         "Expecting following arguments:\n"
         "size of the image (even number)\n"
         "number of images (>=1)\n"
         "batch size (>=1)\n"
         "number of symmetries (>=1)\n"
         "'fast' for using fast interpolation method (other values interpreted as 'precise')\n"
         "'table' for using lookup table for interpolation coefficient (other values interpreted "
         "as 'dynamic computation'\n\n";
  }

  auto size = umpalumpa::data::Size(x, x, 1, n);
  const auto program = []() -> std::unique_ptr<FourierReconstruction<float>> {
     return std::make_unique<FourierReconstructionCUDA<float>>();
    // return std::make_unique<FourierReconstructionCPU<float>>();
  //  return std::make_unique<FourierReconstructionStarPU<float>>();
  }();
  program->Execute(size, symmetries, batch, type, interpolation);
}
