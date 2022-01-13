#pragma once

#include <libumpalumpa/algorithms/basic_algorithm.hpp>
#include <libumpalumpa/data/payload_wrapper.hpp>
#include <libumpalumpa/data/fourier_descriptor.hpp>
#include <libumpalumpa/data/logical_desriptor.hpp>
#include <libumpalumpa/data/payload.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/settings.hpp>

namespace umpalumpa::fourier_reconstruction {

template<typename T, typename U> struct OutputDataWrapper : public data::PayloadWrapper<T, U>
{
  OutputDataWrapper(std::tuple<T, U> &t) : data::PayloadWrapper<T, U>(t) {}
  OutputDataWrapper(T &volume, U &weight) : data::PayloadWrapper<T, U>(volume, weight) {}
  const T &GetVolume() const { return std::get<0>(this->payloads); };
  const U &GetWeight() const { return std::get<1>(this->payloads); };
  typedef T VolumeType;
  typedef U WeightType;
};

template<typename T, typename U> struct InputDataWrapper : public data::PayloadWrapper<T, T, U>
{
  InputDataWrapper(std::tuple<T, T, U> &t) : data::PayloadWrapper<T, T, U>(t) {}
  InputDataWrapper(T &fft, T &volume, U &weight)
    : data::PayloadWrapper<T, T, U>(fft, volume, weight)
  {}
  const T &GetFFT() const { return std::get<0>(this->payloads); };
  const T &GetVolume() const { return std::get<1>(this->payloads); };
  const U &GetWeight() const { return std::get<2>(this->payloads); };
  typedef T VolumeType;
  typedef T FFTType;
  typedef U WeightType;
};

class AFR
  : public BasicAlgorithm<OutputDataWrapper<data::Payload<data::FourierDescriptor>,
                            data::Payload<data::LogicalDescriptor>>,
      InputDataWrapper<data::Payload<data::FourierDescriptor>,
        data::Payload<data::LogicalDescriptor>>,
      Settings>
{
protected:
  bool IsValid(const OutputData &out, const InputData &in, const Settings &) const override
  {
    // TODO add size checks
    auto IsValidAndEqual = [](auto &p1, auto &p2) { return p1.IsValid() && p1 == p2; };
    return IsValidAndEqual(out.GetVolume(), in.GetVolume())
           && IsValidAndEqual(out.GetWeight(), in.GetWeight()) && in.GetFFT().IsValid();
  }
};

}// namespace umpalumpa::fourier_reconstruction