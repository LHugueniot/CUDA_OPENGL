#include "Geometry.cuh"

struct WorldState
{
    const ei::Vector3f m_gravConstant = {0.f, -9.8f, 0.f};
    const float m_timeStep = 1.f / 60.f;
};

struct PBDGeometry : Geometry
{
    /// ===================Inherited from Geometry===================

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

    /// ========================PBD specific=========================

    // Indices of fixed vertices
    uint *d_fixedVertexIdxBufferData = nullptr;
    uint *d_nFixedVertexIdxBufferElems = nullptr;

    // Device buffer containing the length of distance constraints
    float *d_distanceConstraintLengthBufferData = nullptr;
    // Device buffer containing pairs of indices to vertices
    // representing a distance constraint
    uint *d_distanceConstraintsIdxBufferData = nullptr;
    uint d_nDistanceConstraintsIdxBufferElems = 0;

    // Distance constraint sets grouped by color according to a color
    // graphing algorithm
    uint **d_distanceConstraintSets = nullptr;
    uint d_nDistanceConstraintSets = 0;
    uint *d_nDistanceConstraintsPerSet = nullptr;

    // Vertex velocities buffer size will be d_nVertexPositionBufferElems
    float *d_vertexVelocitiesBufferData = nullptr;
    // Num of vertex masse elems in buffer will be d_nVertexPositionBufferElems/3
    float *d_nVertexMassesBufferElems = nullptr;
};

template <typename FixedVertexIxd_t, uint fixedVertexIdxsSize>
PBDGeometry initializePBDParameters(Geometry &geometry,
                                    FixedVertexIxd_t fixedVertexIdxs)
{
    // geometry
}

void stepPBD(PBDGeometry &geometry,
             WorldState &worldState);
