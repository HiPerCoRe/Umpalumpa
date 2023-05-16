#pragma once

#include <iostream>
#include <libumpalumpa/operations/fourier_reconstruction/blob_order.hpp>

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

  bool IsEquivalentTo(const Settings &ref) const
  {
    return interpolation == ref.interpolation && order == ref.order && type == ref.type
           && alpha == ref.alpha && blobRadius == ref.blobRadius;
  }

  void Serialize(std::ostream &out) const
  {
    out << static_cast<int>(interpolation) << ' ' << static_cast<int>(order) << ' '
        << static_cast<int>(type) << ' ' << alpha << ' ' << blobRadius << '\n';
  }

  static auto Deserialize(std::istream &in)
  {
    int i, bo, t;
    float a, br;
    in >> i >> bo >> t >> a >> br;
    Settings s;
    s.SetInterpolation(static_cast<Interpolation>(i));
    s.SetBlobOrder(static_cast<BlobOrder>(bo));
    s.SetType(static_cast<Type>(t));
    s.SetAlpha(a);
    s.SetBlobRadius(br);
    return s;
  }

private:
  Interpolation interpolation = Interpolation::kDynamic;
  BlobOrder order = BlobOrder::k0;
  Type type = Type::kFast;
  float alpha = 15.f;
  float blobRadius = 1.9f;
};
}// namespace umpalumpa::fourier_reconstruction
