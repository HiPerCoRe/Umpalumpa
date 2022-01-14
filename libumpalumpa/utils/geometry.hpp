#pragma once

#include <libumpalumpa/utils/cuda_compatibility.hpp>
#include <libumpalumpa/data/point3d.hpp>

/** Calculates Z coordinate of the point [x, y] on the plane defined by p0 (origin) and normal */
CUDA_HD
template<typename T> T getZ(T x, T y, const Point3D<T> &n, const Point3D<T> &p0)
{
  // from a(x-x0)+b(y-y0)+c(z-z0)=0
  return (-n.x * (x - p0.x) - n.y * (y - p0.y)) / n.z + p0.z;
}

/** Calculates Y coordinate of the point [x, z] on the plane defined by p0 (origin) and normal */
CUDA_HD
template<typename T> T getY(T x, T z, const Point3D<T> &n, const Point3D<T> &p0)
{
  // from a(x-x0)+b(y-y0)+c(z-z0)=0
  return (-n.x * (x - p0.x) - n.z * (z - p0.z)) / n.y + p0.y;
}


/** Calculates X coordinate of the point [y, z] on the plane defined by p0 (origin) and normal */
CUDA_HD
template<typename T> T getX(T y, T z, const Point3D<T> &n, const Point3D<T> &p0)
{
  // from a(x-x0)+b(y-y0)+c(z-z0)=0
  return (-n.y * (y - p0.y) - n.z * (z - p0.z)) / n.x + p0.x;
}