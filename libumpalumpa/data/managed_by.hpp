#pragma once

namespace umpalumpa::data {
enum class ManagedBy {

  /** We don't know and we don't care */
  External,

  /** CUDA unified memory */
  CUDA,

  /** Programmer is responsible for data placement */
  Manually,

  /** Memory is managed by StarPU */
  StarPU,

  /** We don't know, and it might be a problem */
  Unknown
};
}
