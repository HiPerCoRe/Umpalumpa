#include "flexalign_cpu.hpp"
#include "flexalign_cuda.hpp"
#include "flexalign_starpu.hpp"

int main()
{
  auto size = umpalumpa::data::Size(10, 10, 1, 3);

  auto program = FlexAlignStarPU<float>();
  program.Execute(size);
}