#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/blob_order.hpp>

namespace umpalumpa::fourier_reconstruction {

class Settings
{
public:
  enum class Interpolation { kDynamic, kLookup };

  auto GetInterpolation() const { return interpolation; }

  void SetInterpolation(const Interpolation &i) { this->interpolation = i; }

  auto GetBlobOrder() const { return order; }

  void SetBlobOrder(const BlobOrder &o) { this->order = o; }

  auto GetAlpha() const { return alpha; }

  void SetAlpha(float a) { this->alpha = a; }

private:
  Interpolation interpolation = Interpolation::kDynamic;
  BlobOrder order = BlobOrder::k0;
  float alpha = 15.f;
};
}// namespace umpalumpa::fourier_reconstruction