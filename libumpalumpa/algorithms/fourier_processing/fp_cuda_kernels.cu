#include <libumpalumpa/data/size.hpp>

// This kernel assumes that low frequencies are located in the corners

// FIXME not sure to work properly when out dimensions are bigger than
//       input dimensions

__global__
void scaleFFT2DKernel(const float2* __restrict__ in, float2* __restrict__ out,
    umpalumpa::data::Size inSize, umpalumpa::data::Size outSize,
    const float* __restrict__ filter, float normFactor ) {
  // assign pixel to thread
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;

  if (idx >= outSize.x || idy >= outSize.y ) return;
  size_t fIndex = idy*outSize.x + idx; // index within single image
  
  float centerCoef = 1-2*((idx+idy)&1); // center FT, input must be even,
                                        // useful when you will do IFFT of correlation ->
                                        // -> correlation maxima will be in the center of the image
  int yhalf = outSize.y/2;

  size_t origY = (idy <= yhalf) ? idy : (inSize.y - (outSize.y-idy)); // take top N/2+1 and bottom N/2 lines
  for (int n = 0; n < inSize.n; n++) {
    size_t iIndex = n*inSize.x*inSize.y + origY*inSize.x + idx; // index within consecutive images
    size_t oIndex = n*outSize.x*outSize.y + fIndex; // index within consecutive images
    out[oIndex] = in[iIndex];
    if (applyFilter) {
      out[oIndex].x *= filter[fIndex];
      out[oIndex].y *= filter[fIndex];
    }
    if (0 == idx || 0 == idy) {
      out[oIndex] = {0, 0}; // ignore low frequency, this should increase precision a bit
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

