// #include "flexalign_cpu.hpp"
// #include "flexalign_cuda.hpp"
#include "flexalign_starpu.hpp"
#include <starpu.h>
#include <stdlib.h>

int main()
{
  auto size = umpalumpa::data::Size(4096, 4096, 1, 40);

  if (true) {
    static char mem[] = "STARPU_LIMIT_CPU_MEM=20480"; // has to be static
    putenv(mem);
    STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "StarPU init");
    {
      // Program needs to be in extra scope, so that destructor is called
      // before we exit StarPU
      auto program = FlexAlignStarPU<float>();
      program.Execute(size);
    }
    starpu_shutdown();
    // } else {
    //   auto program = FlexAlignCPU<float>();
    //   program.Execute(size);
  }
}
