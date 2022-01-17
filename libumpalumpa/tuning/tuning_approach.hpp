#pragma once

namespace umpalumpa::tuning {

/**
 * Tuning of a strategy is done according to the selected TuningApproach.
 */
enum class TuningApproach {
  kNoTuning,// Tuning is switched off
  kEntireStrategy,// All kernels of the strategy should be tuned
  kSelectedKernels,// Only selected kernels will be tuned, the rest will run with the best known
                   // configuration
};

}// namespace umpalumpa::tuning
