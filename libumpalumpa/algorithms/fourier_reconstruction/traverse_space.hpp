#pragma once

#include <libumpalumpa/data/point3d.hpp>

namespace umpalumpa::fourier_reconstruction {

/**
 * Struct describing a how to best traverse a single projection
 * during the Fourier Reconstruction.
 * It describes an Axis Aligned Bounding Box (AABB) that wraps
 * some projection plane (or cuboid, in case it has some thickness) oriented in the memory
 */
struct TraverseSpace
{
  int minY, minX, minZ;// coordinates of bottom left front corner of the AABB

  int maxY, maxX, maxZ;// coordinates of upper right back corner of the AABB

  float maxDistanceSqr;// max squared distance from the center of the Fourier Space which should be
                       // processed (i.e. 'max frequency to process')
  enum class Direction {
    XY,
    XZ,
    YZ
  } dir;// optimal plane for traversing (i.e. 'process this plane and iterate in last direction')

  /**
   * Projection itself is a plane (with/without some thickness) somehow oriented in the AABB.
   * These variables hold normal to the plane
   */
  data::Point3D<float> unitNormal;

  /**
   * Projection can have some thickness due to the blob radius.
   * These variables hold the origin of the lower/upper plane.
   * Bear in mind that 'lower/upper' refers to initial orientation, before applying projection
   * specific rotation, so it can happen that 'topOrigin' is lower than 'bottomOrigin'.
   * In case the blob radius is zero, these variables hold the same values
   */
  data::Point3D<float> topOrigin, bottomOrigin;

  int projectionIndex;// index to array of projections, which holds appropriate (visual) data

  float transformInv[3][3];// rotation matrix, describing transformation to default position

  float weight;// of the projection
};
}// namespace umpalumpa::fourier_reconstruction
