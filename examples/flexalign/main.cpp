#include "flexalign_cpu.hpp"
// #include "flexalign_cuda.hpp"
#include "flexalign_starpu.hpp"

int main()
{
  auto size = umpalumpa::data::Size(2048, 2048, 1, 40);
  const auto program = []() -> std::unique_ptr<FlexAlign<float>> {
    if (true) { return std::make_unique<FlexAlignStarPU<float>>(); }
    return std::make_unique<FlexAlignCPU<float>>();
  }();
  program->Execute(size);
}
