#pragma once

/**
 * Serves for distinguishing tests that can be run only at some specific processing units.
 */
enum class TestProcessingUnit {
  kAny,
  kCPU,
  kGPU,
  kSTARPU,
};
