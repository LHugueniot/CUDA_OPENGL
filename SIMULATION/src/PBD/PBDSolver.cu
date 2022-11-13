#include "Sim/PBD/PBDSolver.cuh"
#include "Utils/Cuda.cuh"

// PBDGeometry initializePBDGeometryFromHost<FixedVertexIxd_t>(Geometry&
// geometry,
//                                           FixedVertexIxd_t
//                                           fixedVertexIndices);

__device__ float length(ei::Vector3f const &v)
{
    return sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
}

__global__ void calculateEdgeLengths(float *vertexPositionBufferData,
                                     uint nVertexPositionBufferElems,
                                     uint *edgeIdxBufferData,
                                     uint nEdgeIdxBufferElems,
                                     float *distanceConstraintLengthBufferData)
{
    uint threadFlatIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Get the index range we're operating on (i.e first and second vertex indices)
    uint edgeStartIdx = threadFlatIdx * 2;
    uint edgeEndIdx = edgeStartIdx + 1;

    if (!(edgeEndIdx < nEdgeIdxBufferElems))
        return;

    uint vert1StartIdx = edgeIdxBufferData[edgeStartIdx] * 3;
    uint vert1EndIdx = vert1StartIdx + 2;

    uint vert2StartIdx = edgeIdxBufferData[edgeEndIdx] * 3;
    uint vert2EndIdx = vert2StartIdx + 2;

    if (!(vert1EndIdx < nVertexPositionBufferElems) ||
        !(vert2EndIdx < nVertexPositionBufferElems))
        return;

    ei::Map<ei::Vector3f> vert1(&vertexPositionBufferData[vert1StartIdx]);
    ei::Map<ei::Vector3f> vert2(&vertexPositionBufferData[vert2StartIdx]);

    ei::Vector3f diff = (vert2 - vert1);

    distanceConstraintLengthBufferData[threadFlatIdx] = length(diff);
}

bool initializePBDParameters(PBDGeometry &g,
                             uint *fixedVertexIdxs,
                             uint nFixedVertexIdxsElems)
{
    if (!g.d_vertexPositionBufferData || g.d_nVertexPositionBufferElems == 0 ||
        !g.d_edgeIdxBufferData || g.d_nEdgeIdxBufferElems == 0)
    {
        WARNING_MSG("Geometry has not been initialized.");
        return false;
    }

    // Initialize the fixed vertices data
    g.d_nFixedVertexIdxBufferElems = nFixedVertexIdxsElems;
    uint fixedVertexIdxsSize = g.d_nFixedVertexIdxBufferElems * sizeof(uint);
    cutilSafeCall(cudaMalloc(&g.d_fixedVertexIdxBufferData,
                             fixedVertexIdxsSize));
    cutilSafeCall(cudaMemcpy(g.d_fixedVertexIdxBufferData,
                             fixedVertexIdxs,
                             fixedVertexIdxsSize,
                             cudaMemcpyDefault));

    // Initialize the distance contraint length data
    assert(g.d_nEdgeIdxBufferElems % 2 == 0);

    g.d_nDistanceConstraintLengthBufferElems =
        static_cast<uint>(g.d_nEdgeIdxBufferElems / 2);

    uint distanceConstraintLengthBufferSize = g.d_nDistanceConstraintLengthBufferElems * sizeof(float);
    cudaMalloc(&g.d_distanceConstraintLengthBufferData,
               distanceConstraintLengthBufferSize);

    uint nEdges = static_cast<uint>(g.d_nEdgeIdxBufferElems / 2);

    calculateEdgeLengths<<<1, nEdges>>>(g.d_vertexPositionBufferData,
                                        g.d_nVertexPositionBufferElems,
                                        g.d_edgeIdxBufferData,
                                        g.d_nEdgeIdxBufferElems,
                                        g.d_distanceConstraintLengthBufferData);

    //
    g.d_nDistanceConstraintsIdxBufferElems = g.d_nEdgeIdxBufferElems;
    uint distanceConstraintsIdxBufferSize = g.d_nDistanceConstraintsIdxBufferElems * sizeof(uint);

    cutilSafeCall(cudaMalloc(&g.d_distanceConstraintsIdxBufferData,
                             distanceConstraintsIdxBufferSize));

    cutilSafeCall(cudaMemcpy(g.d_distanceConstraintsIdxBufferData,
                             g.d_edgeIdxBufferData,
                             distanceConstraintsIdxBufferSize,
                             cudaMemcpyDefault));

    uint edgeIdxBufferSize = g.d_nEdgeIdxBufferElems * sizeof(uint);
    uint *h_edgeIdxBufferData = new uint[g.d_nEdgeIdxBufferElems];
    cutilSafeCall(cudaMemcpy(h_edgeIdxBufferData,
                             g.d_edgeIdxBufferData,
                             edgeIdxBufferSize,
                             cudaMemcpyDeviceToHost));

    std::vector<uint> edgeIdxBufferData;
    edgeIdxBufferData.resize(g.d_nEdgeIdxBufferElems);
    for (uint i = 0; i < g.d_nEdgeIdxBufferElems; i++)
    {
        edgeIdxBufferData[i] = h_edgeIdxBufferData[i];
    }
    std::cout << edgeIdxBufferData << std::endl;

    // Distance constraint graph
    std::map<std::pair<uint, uint>, std::list<std::pair<uint, uint>>> dcGraph;

    std::cout << "g.d_nEdgeIdxBufferElems: " << g.d_nEdgeIdxBufferElems << std::endl;

    for (uint i = 0; i < g.d_nEdgeIdxBufferElems; i += 2)
    {
        auto currEdge = makeOrderedIdxPair(h_edgeIdxBufferData[i], h_edgeIdxBufferData[i + 1]);

        auto currEdgeItr = dcGraph.find(currEdge);
        if (currEdgeItr == dcGraph.end())
        {
            dcGraph[currEdge] = {};
        }

        for (auto &[edge, adjEdges] : dcGraph)
        {
            auto &[idx1, idx2] = edge;
            if (currEdge.first == idx1 || currEdge.first == idx2 ||
                currEdge.second == idx1 || currEdge.second == idx2)
            {
                dcGraph[currEdge].push_back(edge);
                dcGraph[edge].push_back(currEdge);
            }
        }
    }

    std::cout << dcGraph << std::endl;
}

void stepPBD(PBDGeometry &geometry,
             std::vector<uint> fixedVertexIndices,
             WorldProperties &worldProperties) {}