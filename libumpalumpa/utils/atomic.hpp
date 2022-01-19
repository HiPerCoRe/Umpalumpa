#pragma once

#include <atomic>

namespace umpalumpa::utils {

/** Atomically increments the value pointed at by ptr by value.
 * Uses relaxed memory model with no reordering guarantees. */
void AtomicAddFloat(volatile float *ptr, float addedValue)
{
  static_assert(sizeof(float) == sizeof(uint32_t), "atomicAddFloat requires floats to be 32bit");

  // This is probably fine, since the constructor/destructor should be trivial
  // (As of C++11, this is guaranteed only for integral type specializations, but it is probably
  // reasonably safe to assume that this will hold for floats as well. C++20 requies that by spec.)
  volatile std::atomic<float> &atomicPtr = *reinterpret_cast<volatile std::atomic<float> *>(ptr);
  float current = atomicPtr.load(std::memory_order::memory_order_relaxed);
  while (true) {
    const float newValue = current + addedValue;
    // Since x86 does not allow atomic add of floats (only integers), we have to implement it
    // through CAS
    if (atomicPtr.compare_exchange_weak(
          current, newValue, std::memory_order::memory_order_relaxed)) {
      // Current was still current and was replaced with the newValue. Done.
      return;
    }
    // Comparison failed. current now contains the new value and we try again.
  }
}
}// namespace umpalumpa::utils
