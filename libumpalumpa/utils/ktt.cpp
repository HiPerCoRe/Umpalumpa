#include <libumpalumpa/utils/ktt.hpp>
#include <libumpalumpa/utils/cuda.hpp>

namespace umpalumpa::utils {
ktt::ArgumentMemoryLocation KTTUtils::GetMemoryNode(const umpalumpa::data::PhysicalDescriptor &pd)
{
  using umpalumpa::data::ManagedBy;
  using ktt::ArgumentMemoryLocation;
  switch (pd.GetManager()) {
  case ManagedBy::CUDA:
    return ArgumentMemoryLocation::Unified;
  case ManagedBy::StarPU:
    [[fallthrough]];
  case ManagedBy::Manually:
    return (
      IsGpuPointer(pd.GetPtr()) ? ArgumentMemoryLocation::Device : ArgumentMemoryLocation::Host);
  default:
    break;
  }
  return ArgumentMemoryLocation::Undefined;// because we don't know any better
}
}// namespace umpalumpa::utils
