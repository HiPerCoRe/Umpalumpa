#pragma once
#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/locality.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/direction.hpp>

namespace umpalumpa::fourier_transformation {
class Settings
{
public:
  /**
   * Create Settings for the Fourier Transform
   * You can pick
   *  - the locality (in/out of place)
   *  - the direction (forward / inverse)
   *  - as an exception, you can set number of (CPU) threads for the execution
   * Threads do not affect CUDA implementation.
   * In case of e.g. StarPU, 1 thread will map to one worker
   **/
  Settings(Locality loc, Direction dir, unsigned thr = 1)
    : locality(loc), direction(dir), threads(thr)
  {}

  Locality GetLocality() const { return locality; }
  Direction GetDirection() const { return direction; }

  bool IsForward() const { return direction == Direction::kForward; }
  bool IsOutOfPlace() const { return locality == Locality::kOutOfPlace; }

  int GetVersion() const { return version; }

  Settings CreateInverse() const
  {
    return Settings(locality,
      direction == Direction::kForward ? Direction::kInverse : Direction::kForward,
      threads);
  }

  unsigned GetThreads() const { return threads; }

private:
  static constexpr int version = 1;
  Locality locality;
  Direction direction;
  unsigned threads;
};
}// namespace umpalumpa::fourier_transformation
