#include <libumpalumpa/algorithms/correlation/correlation_cpu.hpp>
#include <gtest/gtest.h>
#include <tests/algorithms/correlation/correlation_tests.hpp>

using namespace umpalumpa::correlation;
using namespace umpalumpa::data;

class CorrelationCPUTest
  : public ::testing::Test
  , public Correlation_Tests
{
public:
  void *Allocate(size_t bytes) override { return malloc(bytes); }

  void Free(void *ptr) override { free(ptr); }

  // CANNOT return "Free" method, because of the destruction order
  FreeFunction GetFree() override
  {
    return [](void *ptr) { free(ptr); };
  }

  Correlation_CPU &GetTransformer() override { return transformer; }

protected:
  Correlation_CPU transformer = Correlation_CPU();
};
#define NAME CorrelationCPUTest
#include <tests/algorithms/correlation/acorrelation_common.hpp>
