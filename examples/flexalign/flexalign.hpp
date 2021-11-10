#pragma once

#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>
#include <memory>

using umpalumpa::data::Size;
using umpalumpa::data::Payload;
using umpalumpa::data::FourierDescriptor;
using umpalumpa::data::LogicalDescriptor;


// FIXME remove
#include <iostream>


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
  /**
   * Generate Payload representing a single image of given size.
   * Payload does not hold any data.
   **/
  std::unique_ptr<Payload<LogicalDescriptor>> Generate(size_t index, const Size &size);

  virtual std::unique_ptr<Payload<FourierDescriptor>>
    ConvertToFFTAndCrop(size_t index, Payload<LogicalDescriptor> &img, const Size &cropSize) = 0;

  std::unique_ptr<Payload<FourierDescriptor>> Correlate(size_t i,
    size_t j,
    Payload<FourierDescriptor> &first,
    Payload<FourierDescriptor> &second);

  Shift FindMax(size_t i, size_t j, Payload<FourierDescriptor> &correlation);

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