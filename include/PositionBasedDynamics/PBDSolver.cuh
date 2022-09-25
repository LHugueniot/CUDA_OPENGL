#include "Geometry.cuh"

struct WorldState
{
    const ei::Vector3f m_gravConstant = {0.f, -9.8f, 0.f};
    const float m_timeStep = 1.f/60.f;
};

struct PBDGeometry : Geometry
{
    /// ===================Inherited from Geometry===================

    // Device buffer containing values for each mesh vertex
    // should contain 3 * nVertices elements
    float* d_vertexPositionBufferData = nullptr;
    // Size of vertex position buffer, should be
    // 3 * nVertices
    uint d_vertexPositionBufferSize = 0;

    // Device buffer containing indices of vertices for each mesh
    // edge, should contain a multiple of 2, elements
    uint* d_edgeIdxBufferData = nullptr;
    // Size of edge indices buffer, should be a 2 * nEdges
    uint d_edgeIdxBufferSize = 0;

    // Device buffer containing indices of edges for each mesh
    // triangle, should contain a multiple of 3, elements
    uint* d_triangleIdxBufferData = nullptr;
    // Size of triangle indices buffer, should be a 3 * nTriangles
    uint d_triangleIdxBufferSize = 0;

    /// ========================PBD specific=========================

    // Indices of fixed vertices
    uint* d_fixedVertexIdxBufferData = nullptr;

    // Device buffer containing the length of distance constraints
    float* d_distanceConstraintLengthBufferData = nullptr;
    // Device buffer containing pairs of indices to vertices
    // representing a distance constraint
    uint* d_distanceConstraintsIdxBufferData = nullptr;
    uint d_distanceConstraintsIdxBufferSize = 0;

    // Distance constraint sets grouped by color according to a color
    // graphing algorithm
    uint** d_distanceConstraintSets = nullptr;
    uint d_distanceConstraintSetNum = 0;
    uint* d_distanceConstraintSetSize = nullptr;

    // Vertex velocities buffer size will be d_vertexPositionBufferSize
    float* d_vertexVelocitiesBufferData = nullptr;
    // Vertex masses buffer size will be d_vertexPositionBufferSize/3
    float* d_vertexMassesBufferData = nullptr;
};

template<typename FixedVertexIxd_t, uint fixedVertexIdxsSize>
PBDGeometry initializePBDParameters(Geometry& geometry,
                                    FixedVertexIxd_t fixedVertexIdxs)
{
    //geometry
}

void stepPBD(PBDGeometry& geometry,
             WorldState& worldState);
