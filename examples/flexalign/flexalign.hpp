#pragma once

#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>

using umpalumpa::data::Size;
using umpalumpa::data::Payload;
using umpalumpa::data::FourierDescriptor;
using umpalumpa::data::LogicalDescriptor;
using umpalumpa::data::PhysicalDescriptor;
using umpalumpa::data::DataType;
using umpalumpa::fourier_transformation::AFFT;
using umpalumpa::fourier_processing::AFP;

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
  Payload<FourierDescriptor>
    ConvertToFFTAndCrop(size_t index, Payload<LogicalDescriptor> &img, Payload<LogicalDescriptor> &filter);
  
  
  /**
   * Generate Payload representing a single image of given size.
   **/
  Payload<LogicalDescriptor> CreatePayloadImage(size_t index, const Size &size);

  /**
   * Create Payload representing a filter applied to the data
   **/
  Payload<LogicalDescriptor> CreatePayloadFilter(const Size &size);

  Payload<FourierDescriptor> CreatePayloadInFFT(size_t index,
    const Payload<LogicalDescriptor> &img);
  Payload<FourierDescriptor> CreatePayloadOutFFT(size_t index,
    const Payload<FourierDescriptor> &inFFT);

  Payload<FourierDescriptor> CreatePayloadInCroppedFFT(size_t index,
    const Payload<FourierDescriptor> &inFFT);
  Payload<FourierDescriptor> CreatePayloadOutCroppedFFT(size_t index, const Size &size);


  //   std::unique_ptr<Payload<FourierDescriptor>> Correlate(size_t i,
  //     size_t j,
  //     Payload<FourierDescriptor> &first,
  //     Payload<FourierDescriptor> &second);

  //   Shift FindMax(size_t i, size_t j, Payload<FourierDescriptor> &correlation);

  /**
   * This method creates a Physical Payload.
   * If necessary, data should be registered in the respective Memory Manager.
   * If tmp is True, this data are not meant for long-term storage.
   **/
  virtual PhysicalDescriptor Create(size_t bytes, DataType type, bool tmp) const = 0;

  virtual void Remove(const PhysicalDescriptor &pd) const = 0;

  DataType GetDataType() const;

  DataType GetComplexDataType() const;

  virtual AFFT &GetFFTAlg() const = 0;

  virtual AFP &GetCropAlg() const = 0;

private:
  /**
   * Generate test image with a cross on specific position
   **/
  void GenerateCross(size_t index,
    const Payload<LogicalDescriptor> &p,
    const Size &crossSize,
    size_t x,
    size_t y);
};