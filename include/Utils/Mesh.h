#ifndef UTIL_MESH_H
#define UTIL_MESH_H

#include <vector>

#include "Utils/Math.h"
#include "Utils/General.h"
#include "Utils/Eigen.h"

void generateLine(std::vector<float> & origin,
                  const ei::Vector3f& position,
                  const float& length,
                  const Dim& direction);

void generateSquare(std::vector<float> & vertices,
                    const ei::Vector3f& origin,
                    const float& squareSize,
                    const std::array<Dim, 2>& dims);

void generateSquarePlane(std::vector<float>& vertices,
                         const ei::Vector3f& origin,
                         const float& squareSize,
                         const std::array<Dim, 2>& dims,
                         const ei::Vector2ui& numSquaresPerDim);

void generateCubeGrid(std::vector<float>& vertices,
                       const ei::Vector3f& origin,
                       const float& cubeSize,
                       const ei::Vector3ui& numCubesPerDim);


#endif /* UTIL_MESH_H */