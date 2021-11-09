#pragma once

#include <libumpalumpa/data/physical_desriptor.hpp>
#include <libumpalumpa/system_includes/ktt.hpp>

namespace umpalumpa::utils {

struct KTTUtils
{
  static ktt::ArgumentMemoryLocation GetMemoryNode(const umpalumpa::data::PhysicalDescriptor &pd);
};
}// namespace umpalumpa::utils
