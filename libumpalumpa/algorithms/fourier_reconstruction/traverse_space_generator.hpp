#pragma once

#include <libumpalumpa/algorithms/fourier_reconstruction/traverse_space.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_common_kernels.hpp>

namespace umpalumpa::fourier_reconstruction {

/**
*          6____5
*         2/___1/
*    +    | |  ||   y
* [0,0,0] |*|7 ||4
*    -    |/___|/  z  sizes are padded with blob-radius
*         3  x  0
* [0,0] is in the middle of the left side (point [2] and [3]), provided the blobSize is 0
* otherwise the values can go to negative values
* origin(point[0]) in in 'high' frequencies so that possible numerical instabilities are moved to high frequencies
*/
static void createProjectionCuboid(data::Point3D<float> (&cuboid)[8], float sizeX, float sizeY, float blobSize) {
	float halfY = sizeY / 2.0f;
	cuboid[3].x = cuboid[2].x = cuboid[7].x = cuboid[6].x = 0.f - blobSize;
	cuboid[0].x = cuboid[1].x = cuboid[4].x = cuboid[5].x = sizeX + blobSize;

	cuboid[3].y = cuboid[0].y = cuboid[7].y = cuboid[4].y = -(halfY + blobSize);
	cuboid[1].y = cuboid[2].y = cuboid[5].y = cuboid[6].y = halfY + blobSize;

	cuboid[3].z = cuboid[0].z = cuboid[1].z = cuboid[2].z = 0.f + blobSize;
	cuboid[7].z = cuboid[4].z = cuboid[5].z = cuboid[6].z = 0.f - blobSize;
}

/** Apply rotation transform to cuboid */
static inline void rotateCuboid(data::Point3D<float> (&cuboid)[8], const float transform[3][3]) {
	for (int i = 0; i < 8; i++) {
		multiply(transform, cuboid[i]);
	}
}

/** Add 'vector' to each element of 'cuboid' */
static inline void translateCuboid(data::Point3D<float> (&cuboid)[8], data::Point3D<float> vector) {
	for (int i = 0; i < 8; i++) {
		cuboid[i].x += vector.x;
		cuboid[i].y += vector.y;
		cuboid[i].z += vector.z;
	}
}

/**
 * Method will calculate Axis Aligned Bound Box of the cuboid and restrict
 * its maximum size
 */
static void computeAABB(data::Point3D<float> (&AABB)[2], data::Point3D<float> (&cuboid)[8],
                        float minX, float minY, float minZ,
                        float maxX, float maxY, float maxZ) {
	AABB[0].x = AABB[0].y = AABB[0].z = std::numeric_limits<float>::max();
	AABB[1].x = AABB[1].y = AABB[1].z = std::numeric_limits<float>::min();
	data::Point3D<float> tmp;
	for (int i = 0; i < 8; i++) {
		tmp = cuboid[i];
		if (AABB[0].x > tmp.x) AABB[0].x = tmp.x;
		if (AABB[0].y > tmp.y) AABB[0].y = tmp.y;
		if (AABB[0].z > tmp.z) AABB[0].z = tmp.z;
		if (AABB[1].x < tmp.x) AABB[1].x = tmp.x;
		if (AABB[1].y < tmp.y) AABB[1].y = tmp.y;
		if (AABB[1].z < tmp.z) AABB[1].z = tmp.z;
	}
	// limit to max size
	if (AABB[0].x < minX) AABB[0].x = minX;
	if (AABB[0].y < minY) AABB[0].y = minY;
	if (AABB[0].z < minZ) AABB[0].z = minZ;
	if (AABB[1].x > maxX) AABB[1].x = maxX;
	if (AABB[1].y > maxY) AABB[1].y = maxY;
	if (AABB[1].z > maxZ) AABB[1].z = maxZ;
}

/**
 * Method calculates a traversal space information for specific projection
 * imgSizeX - X size of the projection
 * imgSizeY - Y size of the projection
 * transform - forward rotation that should be applied to the projection
 * transformInv - inverse transformation
 * space - which will be filled
 */
static void computeTraverseSpace(uint32_t imgSizeX,
  uint32_t imgSizeY,
  const float transform[3][3],
  TraverseSpace &space,
  uint32_t maxVolumeIndexX,
  uint32_t maxVolumeIndexYZ,
  bool useFast,
  float blobRadius,
  float weight)
{
  data::Point3D<float> cuboid[8];
  data::Point3D<float> AABB[2];
  data::Point3D<float> origin = { maxVolumeIndexX / 2.f, maxVolumeIndexYZ / 2.f, maxVolumeIndexYZ / 2.f };
  createProjectionCuboid(cuboid, imgSizeX, imgSizeY, useFast ? 0.f : blobRadius);
  rotateCuboid(cuboid, transform);
  translateCuboid(cuboid, origin);
  computeAABB(AABB, cuboid, 0, 0, 0, maxVolumeIndexX, maxVolumeIndexYZ, maxVolumeIndexYZ);

  // store data
  space.minZ = floor(AABB[0].z);
  space.minY = floor(AABB[0].y);
  space.minX = floor(AABB[0].x);
  space.maxZ = ceil(AABB[1].z);
  space.maxY = ceil(AABB[1].y);
  space.maxX = ceil(AABB[1].x);
  space.topOrigin = cuboid[4];
  space.bottomOrigin = cuboid[0];
  space.maxDistanceSqr =
    (imgSizeX + (useFast ? 0.f : blobRadius)) * (imgSizeX + (useFast ? 0.f : blobRadius));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { space.transformInv[i][j] = transform[j][i]; }
  }

  // calculate best traverse direction
  space.unitNormal.x = space.unitNormal.y = 0.f;
  space.unitNormal.z = 1.f;
  multiply(transform, space.unitNormal);
  float nX = std::abs(space.unitNormal.x);
  float nY = std::abs(space.unitNormal.y);
  float nZ = std::abs(space.unitNormal.z);

  // biggest vector indicates ideal direction
  if (nX >= nY && nX >= nZ) {// iterate YZ plane
    space.dir = TraverseSpace::Direction::YZ;
  } else if (nY >= nX && nY >= nZ) {// iterate XZ plane
    space.dir = TraverseSpace::Direction::XZ;
  } else if (nZ >= nX && nZ >= nY) {// iterate XY plane
    space.dir = TraverseSpace::Direction::XY;
  }

  space.weight = weight;
}

}// namespace umpalumpa::fourier_reconstruction