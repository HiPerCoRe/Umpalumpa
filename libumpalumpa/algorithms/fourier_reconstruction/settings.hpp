#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/blob_order.hpp>

namespace umpalumpa::fourier_reconstruction {

class Settings
{
public:
  enum class Interpolation { kDynamic, kLookup };

  enum class Type { kFast, kPrecise };

  auto GetInterpolation() const { return interpolation; }

  void SetInterpolation(const Interpolation &i) { this->interpolation = i; }

  auto GetBlobOrder() const { return order; }

  void SetBlobOrder(const BlobOrder &o) { this->order = o; }

  auto GetAlpha() const { return alpha; }

  void SetAlpha(float a) { this->alpha = a; }

  auto GetType() const { return type; }

  void SetType(const Type &t) { this->type = t; }

  auto GetBlobRadius() const { return blobRadius; }

  void SetBlobRadius(float r) { this->blobRadius = r; }

private:
  Interpolation interpolation = Interpolation::kDynamic;
  BlobOrder order = BlobOrder::k0;
  Type type = Type::kFast;
  float alpha = 15.f;
  float blobRadius = 1.9f;
};
}// namespace umpalumpa::fourier_reconstruction