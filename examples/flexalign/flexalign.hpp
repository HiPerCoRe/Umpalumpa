#pragma once

#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>
#include <libumpalumpa/algorithms/correlation/acorrelation.hpp>
#include <libumpalumpa/algorithms/extrema_finder/aextrema_finder.hpp>

#include <mutex>

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


protected:
  Payload<FourierDescriptor> ConvertToFFTAndCrop(size_t index,
    Payload<LogicalDescriptor> &img,
    Payload<LogicalDescriptor> &filter);


  Payload<FourierDescriptor> CreatePayloadInFFT(size_t index,
    const Payload<LogicalDescriptor> &img);
  Payload<FourierDescriptor> CreatePayloadOutFFT(size_t index,
    const Payload<FourierDescriptor> &inFFT);

  Payload<FourierDescriptor>
    CreatePayloadOutInverseFFT(size_t i, size_t j, const Payload<FourierDescriptor> &inFFT);

  Payload<FourierDescriptor> CreatePayloadInCroppedFFT(size_t index,
    const Payload<FourierDescriptor> &inFFT);
  Payload<FourierDescriptor> CreatePayloadOutCroppedFFT(size_t index, const Size &size);

  Payload<LogicalDescriptor>
    CreatePayloadLocMax(size_t i, size_t j, const Payload<FourierDescriptor> &correlation);
  Payload<LogicalDescriptor>
    CreatePayloadMaxIn(size_t i, size_t j, const Payload<FourierDescriptor> &correlation);


  /**
   * This method creates a Physical Payload.
   * If necessary, data should be registered in the respective Memory Manager.
   * If tmp is True, this data are not meant for long-term storage.
   **/
  virtual PhysicalDescriptor Create(size_t bytes, DataType type, bool tmp) const = 0;
  virtual PhysicalDescriptor CreatePD(size_t bytes, DataType type, bool copyInRAM) = 0;

  /**
   * This method removes all data allocated by the Physical Descriptor
   **/
  virtual void Remove(const PhysicalDescriptor &pd) const = 0;
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

  virtual void Synchronize() = 0;

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
  Shift Transform(const Shift &s, float scaleX, float scaleY, float shiftX, float shiftY)
  {
    auto x = -(s.x - shiftX) * scaleX;
    auto y = -(s.y - shiftY) * scaleY;
    return { x, y };
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


  Shift FindMax(Payload<FourierDescriptor> &outCorrelation,
  const std::string &name);

  void LogResult(size_t i, size_t j, const Shift &shift);

   std::mutex mutex;

  //   /**
  //  * This method fetches data represented by the Payload to main RAM.
  //  * Once not needed in RAM, data should be released.
  //  **/
  // template<typename P> const Payload<P> &Acquire(const Payload<P> &p) const
  // {
  //   Acquire(p.dataInfo);
  //   return p;
  // }

  // /**
  //  * This method release data represented by the Payload from main RAM.
  //  * It has to be called once data fetched by Acquire are no longer needed.
  //  **/
  // template<typename P> const Payload<P> &Release(const Payload<P> &p) const
  // {
  //   Release(p.dataInfo);
  //   return p;
  // };
};