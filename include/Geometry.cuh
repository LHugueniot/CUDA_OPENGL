#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

#include <iostream>
#include <vector>

#include "Utils/Eigen.h"
#include "Utils/General.h"

struct Geometry
{
    // Geometry();

    // device buffer containing values for each mesh vertex
    float *d_vertexPositionBufferData = nullptr;
    uint d_nVertexPositionBufferElems = 0;

    // device buffer containing indices for each mesh edge
    uint *d_edgeIdxBufferData = nullptr;
    uint d_nEdgeIdxBufferElems = 0;

    // device buffer containing indices of mesh edges for each mesh triangle
    uint *d_triangleIdxBufferData = nullptr;
    uint d_nTriangleIdxBufferElems = 0;
};

void scaleGeom(Geometry &geom, const ei::Vector3f &scale,
               const ei::Vector3f &pivot = {0.f, 0.f, 0.f});

void rotateGeom(Geometry &geom, const ei::Vector3f &axis, const float angle,
                const ei::Vector3f &pivot = {0.f, 0.f, 0.f});

void translateGeom(Geometry &geom, const ei::Vector3f &translation);

void transformGeom(Geometry &geom, const ei::Transform3f &transform);

#endif /* GEOMETRY_CUH */
