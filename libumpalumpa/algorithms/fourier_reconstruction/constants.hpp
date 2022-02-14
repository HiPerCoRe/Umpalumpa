#pragma once

namespace umpalumpa {
namespace fourier_reconstruction {

  struct Constants
  {
    int cMaxVolumeIndexX;
    int cMaxVolumeIndexYZ;
    float cBlobRadius;
    float cOneOverBlobRadiusSqr;
    float cBlobAlpha;
    float cIw0;
    float cIDeltaSqrt;
    float cOneOverBessiOrderAlpha;
  };

}// namespace fourier_reconstruction
}// namespace umpalumpa