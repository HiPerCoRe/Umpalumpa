#include <libumpalumpa/data/size.hpp>
#include <libumpalumpa/operations/fourier_processing/fp_common_kernels.hpp>

// This kernel assumes that low frequencies are located in the corners

// FIXME not sure to work properly when out dimensions are bigger than
//       input dimensions

template<bool applyFilter, bool normalize, bool center, bool cropFreq, bool shift>
__global__ void scaleFFT2DKernel(const float2 *__restrict__ in,
  float2 *__restrict__ out,
  umpalumpa::data::Size inSize,
  umpalumpa::data::Size inSpatialSize,
  umpalumpa::data::Size outSize,
  const float *__restrict__ filter,
  float normFactor,
  float maxFreqSquare)
{
  // assign pixel to thread
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= outSize.x || idy >= outSize.y) return;
  
  float centerCoef =
  1 - 2 * ((idx + idy) & 1);// center FT, input must be even,
  // useful when you will do IFFT of correlation ->
  // -> correlation maxima will be in the center of the image
  int yhalf = outSize.y / 2;
  
  size_t origY =
  (idy <= yhalf) ? idy : (inSize.y - (outSize.y - idy));// take top N/2+1 and bottom N/2 lines
  
  // compute proper y index in case we should shift coefficients
  size_t tmp = idy + yhalf;
  size_t shiftedY = tmp >= outSize.y ? tmp - outSize.y : tmp;

  // index within single image
  size_t fIndex = (shift ? shiftedY : idy) * outSize.x + idx;
  
  for (int n = 0; n < inSize.n; n++) {
    size_t iIndex =
      n * inSize.x * inSize.y + origY * inSize.x + idx;// index within consecutive images
    size_t oIndex = n * outSize.x * outSize.y + fIndex;// index within consecutive images
    float2 freq = { umpalumpa::fourier_processing::Idx2Freq(idx, inSpatialSize.x), 
      umpalumpa::fourier_processing::Idx2Freq(origY, inSpatialSize.y) };
    out[oIndex] = (cropFreq && (freq.x * freq.x + freq.y * freq.y > maxFreqSquare)) ? make_float2(0, 0) : in[iIndex];
    if (applyFilter) {
      out[oIndex].x *= filter[fIndex];
      out[oIndex].y *= filter[fIndex];
    }
    if (normalize) {
      out[oIndex].x *= normFactor;
      out[oIndex].y *= normFactor;
    }
    if (center) {
      out[oIndex].x *= centerCoef;
      out[oIndex].y *= centerCoef;
    }
  }
}

