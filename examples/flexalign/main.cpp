#include "flexalign_cpu.hpp"
#include "flexalign_cuda.hpp"
#ifdef USE_STARPU
#include "flexalign_starpu.hpp"
#endif
#include <iostream>

int main(int argc, char **argv)
{
  size_t size_x = 4096;
  size_t size_y = 4096;
  size_t frames = 40;
  size_t batch = 5;
  size_t num_of_movies = 1;
  size_t cropped_x = 972;
  size_t cropped_y = 972;

  if (argc == 8) {
    size_x = atoi(argv[1]);
    size_y = atoi(argv[2]);
    frames = atoi(argv[3]);
    batch = atoi(argv[4]);
    num_of_movies = atoi(argv[5]);
    cropped_x = atoi(argv[6]);
    cropped_y = atoi(argv[7]);
  } else {
    std::cout
      << "Using default values.\n\n"
         "Expecting following arguments:\n"
         "X size of the image (already cropped, i.e. 3584 for Xmipp FlexAlign movie of size 3838)\n"
         "Y size of the image (already cropped, i.e. 3584 for Xmipp FlexAlign movie of size 3710)\n"
         "number of images / frames (>=1)\n"
         "batch size (>=1, frames % batch == 0)\n"
         "number of movies (>=1)\n"
         "X size of the image for correlation (already cropped, i.e. 864 for Xmipp FlexAlign movie "
         "of size 3838)\n"
         "Y size of the image for correaltion (already cropped, i.e. 864 for Xmipp FlexAlign movie "
         "of size 3710)\n\n";
  }

  // auto size = umpalumpa::data::Size(2048, 2048, 1, 40);
  auto movieSize = umpalumpa::data::Size(size_x, size_y, 1, frames);
  auto croppedSize = umpalumpa::data::Size(cropped_x, cropped_y, 1, 1);
  const auto program = []() -> std::unique_ptr<FlexAlign<float>> {
//    if (true) { return std::make_unique<FlexAlignStarPU<float>>(); }
    return std::make_unique<FlexAlignCPU<float>>();
  }();
  program->Execute(movieSize, batch, num_of_movies, croppedSize);
}
