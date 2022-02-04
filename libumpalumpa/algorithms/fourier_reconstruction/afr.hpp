#pragma once

#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>
#include <libumpalumpa/data/logical_desriptor.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/settings.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/constants.hpp>

namespace umpalumpa::fourier_reconstruction {

template<typename T, typename U> struct OutputDataWrapper : public data::PayloadWrapper<T, U>
{
  OutputDataWrapper(std::tuple<T, U> &t) : data::PayloadWrapper<T, U>(t) {}
  OutputDataWrapper(T &volume, U &weight) : data::PayloadWrapper<T, U>(volume, weight) {}
  const T &GetVolume() const { return std::get<0>(this->payloads); };
  const U &GetWeight() const { return std::get<1>(this->payloads); };
};

template<typename T, typename U>
struct InputDataWrapper : public data::PayloadWrapper<T, T, U, U, U>
{
  InputDataWrapper(std::tuple<T, T, U, U, U> &t) : data::PayloadWrapper<T, T, U, U, U>(t) {}
  InputDataWrapper(T &fft, T &volume, U &weight, U &traverseSpace, U &blobTable)
    : data::PayloadWrapper<T, T, U, U, U>(fft, volume, weight, traverseSpace, blobTable)
  {}
  const T &GetFFT() const { return std::get<0>(this->payloads); };
  const T &GetVolume() const { return std::get<1>(this->payloads); };
  const U &GetWeight() const { return std::get<2>(this->payloads); };
  const U &GetTraverseSpace() const { return std::get<3>(this->payloads); };
  const U &GetBlobTable() const { return std::get<4>(this->payloads); };
};

class AFR
  : public BasicAlgorithm<OutputDataWrapper<data::Payload<data::FourierDescriptor>,
                            data::Payload<data::LogicalDescriptor>>,
      InputDataWrapper<data::Payload<data::FourierDescriptor>,
        data::Payload<data::LogicalDescriptor>>,
      Settings>
{
public:
  static Constants CreateConstants(const AFR::InputData &in, const Settings &s);
  static void FillBlobTable(const AFR::InputData &in, const Settings &s);

protected:
  bool IsValid(const OutputData &out, const InputData &in, const Settings &) const override
  {
    // TODO add size checks
    auto IsValidAndEqual = [](auto &p1, auto &p2) { return p1.IsValid() && p1 == p2; };
    return IsValidAndEqual(out.GetVolume(), in.GetVolume())
           && IsValidAndEqual(out.GetWeight(), in.GetWeight()) && in.GetFFT().IsValid()
           && in.GetTraverseSpace().IsValid();
  }
};

}// namespace umpalumpa::fourier_reconstruction