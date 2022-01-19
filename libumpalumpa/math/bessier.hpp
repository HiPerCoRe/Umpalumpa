#pragma once

#include <libumpalumpa/math/pi.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/blob_order.hpp>

namespace umpalumpa::math {

double bessi0(double x)
{
  double y, ax, ans;
  if ((ax = fabs(x)) < 3.75) {
    y = x / 3.75;
    y *= y;
    ans = 1.0
          + y
              * (3.5156229
                 + y
                     * (3.0899424
                        + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
  } else {
    y = 3.75 / ax;
    ans =
      (exp(ax) / sqrt(ax))
      * (0.39894228
         + y
             * (0.1328592e-1
                + y
                    * (0.225319e-2
                       + y
                           * (-0.157565e-2
                              + y
                                  * (0.916281e-2
                                     + y
                                         * (-0.2057706e-1
                                            + y
                                                * (0.2635537e-1
                                                   + y * (-0.1647633e-1 + y * 0.392377e-2))))))));
  }
  return ans;
}

double bessi1(double x)
{
  double ax, ans;
  double y;
  if ((ax = fabs(x)) < 3.75) {
    y = x / 3.75;
    y *= y;
    ans = ax
          * (0.5
             + y
                 * (0.87890594
                    + y
                        * (0.51498869
                           + y
                               * (0.15084934
                                  + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))));
  } else {
    y = 3.75 / ax;
    ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2));
    ans = 0.39894228
          + y
              * (-0.3988024e-1
                 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
    ans *= (exp(ax) / sqrt(ax));
  }
  return x < 0.0 ? -ans : ans;
}

double bessi0_5(double x) { return (x == 0) ? 0 : std::sqrt(2 / (PI<double> * x)) * sinh(x); }
template<typename T>
T bessi1_5(T x)
{
  return (x == 0) ? 0 : static_cast<T>(std::sqrt(T(2) / (PI<T> * x)) * (std::cosh(x) - std::sinh(x) / x));
}
double bessi2(double x) { return (x == 0) ? 0 : bessi0(x) - ((2 * 1) / x) * bessi1(x); }
double bessi2_5(double x) { return (x == 0) ? 0 : bessi0_5(x) - ((2 * 1.5) / x) * bessi1_5(x); }
double bessi3(double x) { return (x == 0) ? 0 : bessi1(x) - ((2 * 2) / x) * bessi2(x); }
double bessi3_5(double x) { return (x == 0) ? 0 : bessi1_5(x) - ((2 * 2.5) / x) * bessi2_5(x); }
double bessi4(double x) { return (x == 0) ? 0 : bessi2(x) - ((2 * 3) / x) * bessi3(x); }
template<typename T>
T bessj1_5(T x)
{
    double rj, ry, rjp, ryp;
    bessjy(x, 1.5, &rj, &ry, &rjp, &ryp);
    return rj;
}

float getBessiOrderAlpha(fourier_reconstruction::BlobOrder order, float alpha)
{
  using fourier_reconstruction::BlobOrder;
  switch (order) {
  case BlobOrder::k0:
    return static_cast<float>(bessi0(alpha));
  case BlobOrder::k1:
    return static_cast<float>(bessi1(alpha));
  case BlobOrder::k2:
    return static_cast<float>(bessi2(alpha));
  case BlobOrder::k3:
    return static_cast<float>(bessi3(alpha));
  case BlobOrder::k4:
    return static_cast<float>(bessi4(alpha));
  default:
    return std::numeric_limits<double>::quiet_NaN();
  }
}

}// namespace umpalumpa::math