#include "flexalign_cpu.hpp"
// #include "flexalign_cuda.hpp"
#include "flexalign_starpu.hpp"
#include <starpu.h>

int main()
{
  auto size = umpalumpa::data::Size(16, 16, 1, 3);

  if (false) {
    STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "StarPU init");
    {
      // Program needs to be in extra scope, so that destructor is called
      // before we exit StarPU
      auto program = FlexAlignStarPU<float>();
      program.Execute(size);
    }
    starpu_shutdown();
  }

  if (true) {
    auto program = FlexAlignCPU<float>();
    program.Execute(size);
  }
}
