//template<typename T, typename U, bool applyFilter, bool normalize, bool center>
//__global__
//void scaleFFT2DKernel(const T* __restrict__ in, T* __restrict__ out,
//    int noOfImages, size_t inX, size_t inY, size_t outX, size_t outY,
//    const U* __restrict__ filter, U normFactor) {

// This kernel is assumes that low frequencies are located in the corners

// FIXME not sure to work properly when out dimensions are bigger than
//       input dimensions
extern "C" __global__
void scaleFFT2DKernel(const float2* __restrict__ in, float2* __restrict__ out,
    int noOfImages, size_t inX, size_t inY, size_t outX, size_t outY,
    const float* __restrict__ filter, float normFactor ) {
  // assign pixel to thread
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;

  if (idx >= outX || idy >= outY ) return;
  size_t fIndex = idy*outX + idx; // index within single image
  float filterCoef = filter[fIndex];
  float centerCoef = 1-2*((idx+idy)&1); // center FT, input must be even,
                                        // useful when you will do IFFT of correlation ->
                                        // -> correlation maxima will be in the center of the image
  int yhalf = outY/2;

  size_t origY = (idy <= yhalf) ? idy : (inY - (outY-idy)); // take top N/2+1 and bottom N/2 lines
  for (int n = 0; n < noOfImages; n++) {
    size_t iIndex = n*inX*inY + origY*inX + idx; // index within consecutive images
    size_t oIndex = n*outX*outY + fIndex; // index within consecutive images
    out[oIndex] = in[iIndex];
    if (applyFilter) {
      out[oIndex] *= filterCoef;
    }
    if (0 == idx || 0 == idy) {
      out[oIndex] = {0, 0}; // ignore low frequency, this should increase precision a bit
    }
    if (normalize) {
      out[oIndex] *= normFactor;
    }
    if (center) {
      out[oIndex] *= centerCoef;
    }
  }
}

