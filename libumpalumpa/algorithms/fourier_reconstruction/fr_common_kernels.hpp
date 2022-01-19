#pragma once

#include <libumpalumpa/utils/cuda_compatibility.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/constants.hpp>
#include <libumpalumpa/data/point3d.hpp>
#include <math.h>

namespace umpalumpa {
namespace fourier_reconstruction {

  /** Do 3x3 x 1x3 matrix-vector multiplication */
  CUDA_HD
  template<typename T> void multiply(const T transform[3][3], data::Point3D<T> &inOut)
  {
    T tmp0 = transform[0][0] * inOut.x + transform[0][1] * inOut.y + transform[0][2] * inOut.z;
    T tmp1 = transform[1][0] * inOut.x + transform[1][1] * inOut.y + transform[1][2] * inOut.z;
    T tmp2 = transform[2][0] * inOut.x + transform[2][1] * inOut.y + transform[2][2] * inOut.z;
    inOut.x = tmp0;
    inOut.y = tmp1;
    inOut.z = tmp2;
  }

  CUDA_HD
  float bessi0Fast(float x)
  {// X must be <= 15
    // stable rational minimax approximations to the modified bessel functions, blair, edwards
    // from table 5
    float x2 = x * x;
    float num = -0.8436825781374849e-19f;// p11
    num = fmaf(num, x2, -0.93466495199548700e-17f);// p10
    num = fmaf(num, x2, -0.15716375332511895e-13f);// p09
    num = fmaf(num, x2, -0.42520971595532318e-11f);// p08
    num = fmaf(num, x2, -0.13704363824102120e-8f);// p07
    num = fmaf(num, x2, -0.28508770483148419e-6f);// p06
    num = fmaf(num, x2, -0.44322160233346062e-4f);// p05
    num = fmaf(num, x2, -0.46703811755736946e-2f);// p04
    num = fmaf(num, x2, -0.31112484643702141e-0f);// p03
    num = fmaf(num, x2, -0.11512633616429962e+2f);// p02
    num = fmaf(num, x2, -0.18720283332732112e+3f);// p01
    num = fmaf(num, x2, -0.75281108169006924e+3f);// p00

    float den = 1.f;// q01
    den = fmaf(den, x2, -0.75281109410939403e+3f);// q00

    return num / den;
  }

  CUDA_HD
  float bessi0(float x)
  {
    float y, ax, ans;
    if ((ax = fabsf(x)) < 3.75f) {
      y = x / 3.75f;
      y *= y;
      ans = 1.f
            + y
                * (3.5156229f
                   + y
                       * (3.0899424f
                          + y
                              * (1.2067492f
                                 + y * (0.2659732f + y * (0.360768e-1f + y * 0.45813e-2f)))));
    } else {
      y = 3.75f / ax;
      ans = (expf(ax) * (1 / sqrtf(ax)))
            * (0.39894228f
               + y
                   * (0.1328592e-1f
                      + y
                          * (0.225319e-2f
                             + y
                                 * (-0.157565e-2f
                                    + y
                                        * (0.916281e-2f
                                           + y
                                               * (-0.2057706e-1f
                                                  + y
                                                      * (0.2635537e-1f
                                                         + y
                                                             * (-0.1647633e-1f
                                                                + y * 0.392377e-2f))))))));
    }
    return ans;
  }

  CUDA_HD
  float bessi1(float x)
  {
    float ax, ans;
    float y;
    if ((ax = fabsf(x)) < 3.75f) {
      y = x / 3.75f;
      y *= y;
      ans =
        ax
        * (0.5f
           + y
               * (0.87890594f
                  + y
                      * (0.51498869f
                         + y
                             * (0.15084934f
                                + y * (0.2658733e-1f + y * (0.301532e-2f + y * 0.32411e-3f))))));
    } else {
      y = 3.75f / ax;
      ans = 0.2282967e-1f + y * (-0.2895312e-1f + y * (0.1787654e-1f - y * 0.420059e-2f));
      ans = 0.39894228f
            + y
                * (-0.3988024e-1f
                   + y * (-0.362018e-2f + y * (0.163801e-2f + y * (-0.1031555e-1f + y * ans))));
      ans *= (expf(ax) * (1 / sqrtf(ax)));
    }
    return x < 0.f ? -ans : ans;
  }

  CUDA_HD
  float bessi2(float x) { return (x == 0) ? 0 : bessi0(x) - ((2 * 1) / x) * bessi1(x); }

  CUDA_HD
  float bessi3(float x) { return (x == 0) ? 0 : bessi1(x) - ((2 * 2) / x) * bessi2(x); }

  CUDA_HD
  float bessi4(float x) { return (x == 0) ? 0 : bessi2(x) - ((2 * 3) / x) * bessi3(x); }

  template<int order> CUDA_HD float kaiserValue(float r, float a, const Constants &c)
  {
    float w;
    float rda = r / a;
    if (rda <= 1.f) {
      float rdas = rda * rda;
      float arg = c.cBlobAlpha * sqrtf(1.f - rdas);
      if (order == 0) {
        w = bessi0(arg) * c.cOneOverBessiOrderAlpha;
      } else if (order == 1) {
        w = sqrtf(1.f - rdas);
        w *= bessi1(arg) * c.cOneOverBessiOrderAlpha;
      } else if (order == 2) {
        w = sqrtf(1.f - rdas);
        w = w * w;
        w *= bessi2(arg) * c.cOneOverBessiOrderAlpha;
      } else if (order == 3) {
        w = sqrtf(1.f - rdas);
        w = w * w * w;
        w *= bessi3(arg) * c.cOneOverBessiOrderAlpha;
      } else if (order == 4) {
        w = sqrtf(1.f - rdas);
        w = w * w * w * w;
        w *= bessi4(arg) * c.cOneOverBessiOrderAlpha;
      } else {
        printf("order (%d) out of range in kaiser_value(): %s, %d\n", order, __FILE__, __LINE__);
        w = 0.f;
      }
    } else
      w = 0.f;

    return w;
  }

  CUDA_HD
  float kaiserValueFast(float distSqr, const Constants &c)
  {
    float arg =
      c.cBlobAlpha
      * sqrtf(1.f - (distSqr * c.cOneOverBlobRadiusSqr));// alpha * sqrt(1-(dist/blobRadius^2))
    return bessi0Fast(arg) * c.cOneOverBessiOrderAlpha * c.cIw0;
  }

}// namespace fourier_reconstruction
}// namespace umpalumpa