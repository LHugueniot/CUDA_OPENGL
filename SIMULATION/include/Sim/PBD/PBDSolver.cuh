#include "Geometry/Geometry.cuh"
#include "Utils/General.h"
#include <cstddef>

using std::byte;

struct WorldProperties
{
    ei::Vector3f m_gravConstant = {0.f, -9.8f, 0.f};
    float m_timeStep = 1.f / 120.f;
};

struct PBDGeometry : public Geometry
{
    /// ===================Inherited from Geometry===================
    /*
    // Maybe should enable this later?
    // uint d_nVertexPositions = 0;

    // Device buffer containing values for each mesh vertex
    // should contain 3 * nVertices elements
    float *d_vertexPositionBufferData = nullptr;
    // Number of vertex position elems in buffer, should be
    // 3 * nVertices
    uint d_nVertexPositionBufferElems = 0;

    // Device buffer containing indices of vertices for each mesh
    // edge, should contain a multiple of 2, elements
    uint *d_edgeIdxBufferData = nullptr;
    // Number of edge indices elems in buffer, should be a 2 * nEdges
    uint d_nEdgeIdxBufferElems = 0;

    // Device buffer containing indices of edges for each mesh
    // triangle, should contain a multiple of 3, elements
    uint *d_triangleIdxBufferData = nullptr;
    // Number of triangle indices elems in buffer, should be a 3 * nTriangles
    uint d_nTriangleIdxBufferElems = 0;
    */
    /// ========================PBD specific=========================

    struct PBDData
    {
        // A tightly packed buffer of booleans, packed as chars
        byte *d_isVertexFixedBuffer = nullptr;
        uint d_nIsVertexFixedBufferElems = 0;

        // Indices of fixed vertices
        uint *d_fixedVertexIdxBufferData = nullptr;
        uint d_nFixedVertexIdxBufferElems = 0;

        // Device buffer containing pairs of indices to vertices
        // representing a distance constraint
        uint *d_distanceConstraintsIdxBufferData = nullptr;
        uint d_nDistanceConstraintsIdxBufferElems = 0;

        // Distance constraint sets grouped by color according to a color
        // graphing algorithm
        uint d_nDistanceConstraintSets = 0;
        uint d_maxNDistanceConstraintsInSet = 0;
        // Device buffer containing the indices for a distance constraints
        uint **d_distanceConstraintSets = nullptr;
        uint *d_nDistanceConstraintsPerSet = nullptr;
        // Device buffer containing the rest length of distance constraints
        float **d_dcRestLengthSets = nullptr;
        uint *d_nDcRestLengthPerSet = 0;

        // Vertex velocities buffer size will be d_nVertexPositionBufferElems
        float *d_vertexVelocitiesBufferData = nullptr;
        // Num of vertex masse elems in buffer will be d_nVertexPositionBufferElems/3
        float *d_vertexMassesBufferData = nullptr;
    } pbdData;
};

bool initializePBDParameters(PBDGeometry &g,
                             uint *fixedVertexIdxs,
                             uint nFixedVertexIdxsElems);

// void runPBDSolver(PBDGeometry &g, WorldProperties &worldProperties);
void runPBDSolver(PBDGeometry &g);

void applyExternalForces(PBDGeometry &g, WorldProperties &worldProperties);

template <typename T>
__host__ __device__ void setBoolFromPackedBuffer(T *buffer, uint nBufferElems, uint bufferIdx, bool val)
{
    size_t typeSize = sizeof(T);
    size_t typeSizeInBits = (typeSize * BYTE_BITS);

    uint bufferElemIndex = bufferIdx / typeSizeInBits;

#ifdef DEBUG
#ifndef __CUDA__ARCH__
    assert(!(bufferElemIndex > nBufferElems));
#endif
#endif

    uint bufferElemBitOffset = bufferIdx % typeSizeInBits;
    T &bufferElem = buffer[bufferElemIndex];
    bufferElem |= byte(val << bufferElemBitOffset);
}

template <typename T>
__host__ __device__ bool getBoolFromPackedBuffer(T *buffer, uint nBufferElems, uint bufferIdx)
{
    size_t typeSize = sizeof(T);
    size_t typeSizeInBits = (typeSize * BYTE_BITS);

    uint bufferElemIndex = bufferIdx / typeSizeInBits;

#ifdef DEBUG
#ifndef __CUDA__ARCH__
    assert(!(bufferElemIndex > nBufferElems));
#endif
#endif

    uint bufferElemBitOffset = bufferIdx % typeSizeInBits;
    T &bufferElem = buffer[bufferElemIndex];
    return bool(bufferElem & byte(1 << bufferElemBitOffset));
}
