#include "fr_cpu.hpp"

int main(int argc, char **argv)
{
  size_t x = 7;
  size_t n = 1;
  size_t batch = 1;
  size_t symmetries = 1;
  if (argc == 5) {
    x = atoi(argv[1]);
    n = atoi(argv[2]);
    batch = atoi(argv[3]);
    symmetries = atoi(argv[4]);
  }

  auto size = umpalumpa::data::Size(x, x, 1, n);
  const auto program = []() -> std::unique_ptr<FourierReconstruction<float>> {
    return std::make_unique<FourierReconstructionCPU<float>>();
  }();
  program->Execute(size, symmetries, batch);
}
