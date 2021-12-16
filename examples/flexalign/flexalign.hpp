#pragma once

#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>
#include <libumpalumpa/algorithms/correlation/acorrelation.hpp>
#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>

using umpalumpa::data::Size;
using umpalumpa::data::Payload;
using umpalumpa::data::FourierDescriptor;
using umpalumpa::data::LogicalDescriptor;
using umpalumpa::data::PhysicalDescriptor;
using umpalumpa::data::DataType;
using umpalumpa::fourier_transformation::AFFT;
using umpalumpa::fourier_processing::AFP;
using umpalumpa::correlation::ACorrelation;
using umpalumpa::extrema_finder::AExtremaFinder;

/**
 * This example simulates core functionality of the FlexAlign.
 * In a nutshell, this program takes several 2D images, computer cross-correlation
 * between each pair and outputs their relative shifts.
 * See https://doi.org/10.3390/electronics9061040 for more details
 **/
template<typename T> class FlexAlign
{

public:
  struct Shift
  {
    float x;
    float y;
  };

  void Execute(const Size &size);

  virtual ~FlexAlign() = default;

protected:
  /**
   * This method creates a Physical Payload.
   * If necessary, data should be registered in the respective Memory Manager.
   * If tmp is True, this data are not meant for long-term storage.
   **/
  virtual PhysicalDescriptor CreatePD(size_t bytes, DataType type, bool copyInRAM) = 0;

  /**
   * This method removes all data allocated by the Physical Descriptor
   **/
  virtual void RemovePD(const PhysicalDescriptor &pd) const = 0;

  /**
   * This method fetches data represented by the Physical Descriptor to main RAM.
   * Once not needed in RAM, data should be released.
   **/
  virtual void Acquire(const PhysicalDescriptor &pd) const = 0;

  /**
   * This method release data represented by the Physical Descriptor from main RAM.
   * It has to be called once data fetched by Acquire are no longer needed.
   **/
  virtual void Release(const PhysicalDescriptor &p) const = 0;

  constexpr DataType GetDataType() const;

  constexpr DataType GetComplexDataType() const;

  virtual AFFT &GetForwardFFTAlg() const = 0;

  virtual AFFT &GetInverseFFTAlg() const = 0;

  virtual AFP &GetCropAlg() const = 0;

  virtual AExtremaFinder &GetFindMaxAlg() const = 0;

  virtual ACorrelation &GetCorrelationAlg() const = 0;

private:
  /**
   * Generate test image with a cross on specific position
   **/
  void GenerateClockArms(size_t index,
    const Payload<LogicalDescriptor> &p,
    const Size &armSize,
    size_t posX,
    size_t posY);

  /**
   * Move Shift values by specified offset and scale by the specified factor
   **/
  std::vector<Shift> Transform(const std::vector<Shift> &shift,
    float scaleX,
    float scaleY,
    float shiftX,
    float shiftY)
  {
    auto res = std::vector<Shift>();
    for (const auto &s : shift) {
      auto x = -(s.x - shiftX) * scaleX;
      auto y = -(s.y - shiftY) * scaleY;
      res.push_back({ x, y });
    }
    return res;
  }

  /**
   * Create Payload representing a filter applied to the data
   **/
  Payload<LogicalDescriptor> CreatePayloadFilter(const Size &size);

  /**
   * Generate Payload representing (multiple) image(s) of given size.
   **/
  Payload<LogicalDescriptor> CreatePayloadImage(const Size &size, const std::string &name);

  Payload<FourierDescriptor> ConvertToFFT(const Payload<LogicalDescriptor> &img,
    const std::string &name);

  Payload<FourierDescriptor> Crop(const Payload<FourierDescriptor> &fft,
    Payload<LogicalDescriptor> &filter,
    const std::string &name);

  Payload<FourierDescriptor> Correlate(Payload<FourierDescriptor> &first,
    Payload<FourierDescriptor> &second,
    const std::string &name);

  Payload<FourierDescriptor> ConvertFromFFT(Payload<FourierDescriptor> &correlation,
    const std::string &name);

  Payload<LogicalDescriptor> FindMax(Payload<FourierDescriptor> &outCorrelation,
    const std::string &name);

  std::vector<Shift> ExtractShift(const Payload<LogicalDescriptor> &shift);

  void LogResult(size_t i, size_t j, size_t batch, const std::vector<Shift> &shift);

  size_t NoOfCorrelations(size_t batch, bool isWithin)
  {
    // Note: we are wasting some performence by computing intra-buffer correlations
    // return isWithin ? ((batch * (batch - 1)) / 2) : (batch * batch);
    return batch * batch;
  }

  size_t NoOfBatches(const Size &s, size_t batch)
  {
    return ((s.n / batch + 1) * (s.n / batch)) / 2;
  }
};