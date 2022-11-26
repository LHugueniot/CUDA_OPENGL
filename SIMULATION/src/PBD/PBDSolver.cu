#include "Sim/PBD/PBDSolver.cuh"
#include "Utils/Cuda.cuh"

// PBDGeometry initializePBDGeometryFromHost<FixedVertexIxd_t>(Geometry&
// geometry,
//                                           FixedVertexIxd_t
//                                           fixedVertexIndices);

//__device__ float length(ei::Vector3f const &v)
//{
//    return sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
//}

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

    distanceConstraintLengthBufferData[threadFlatIdx] = diff.norm();
}

bool initializePBDParameters(PBDGeometry &g,
                             uint *fixedVertexIdxs,
                             uint nFixedVertexIdxsElems)
{
    PBDGeometry::PBDData &pd = g.pbdData;
    if (!g.d_vertexPositionBufferData || g.d_nVertexPositionBufferElems == 0 ||
        !g.d_edgeIdxBufferData || g.d_nEdgeIdxBufferElems == 0)
    {
        WARNING_MSG("Geometry has not been initialized.");
        return false;
    }

    // Initialize the fixed vertices data
    pd.d_nFixedVertexIdxBufferElems = nFixedVertexIdxsElems;
    size_t fixedVertexIdxsSize = pd.d_nFixedVertexIdxBufferElems * sizeof(uint);
    cutilSafeCall(cudaMalloc(&pd.d_fixedVertexIdxBufferData,
                             fixedVertexIdxsSize));
    cutilSafeCall(cudaMemcpy(pd.d_fixedVertexIdxBufferData,
                             fixedVertexIdxs,
                             fixedVertexIdxsSize,
                             cudaMemcpyDefault));

    // Initialize the distance contraint length data
    assert(g.d_nEdgeIdxBufferElems % 2 == 0);

    pd.d_nDistanceConstraintLengthBufferElems =
        static_cast<uint>(g.d_nEdgeIdxBufferElems / 2);

    size_t distanceConstraintLengthBufferSize = pd.d_nDistanceConstraintLengthBufferElems * sizeof(float);
    cudaMalloc(&pd.d_distanceConstraintLengthBufferData,
               distanceConstraintLengthBufferSize);

    uint nEdges = static_cast<uint>(g.d_nEdgeIdxBufferElems / 2);

    calculateEdgeLengths<<<1, nEdges>>>(g.d_vertexPositionBufferData,
                                        g.d_nVertexPositionBufferElems,
                                        g.d_edgeIdxBufferData,
                                        g.d_nEdgeIdxBufferElems,
                                        pd.d_distanceConstraintLengthBufferData);

    //
    pd.d_nDistanceConstraintsIdxBufferElems = g.d_nEdgeIdxBufferElems;
    size_t distanceConstraintsIdxBufferSize = pd.d_nDistanceConstraintsIdxBufferElems * sizeof(uint);

    cutilSafeCall(cudaMalloc(&pd.d_distanceConstraintsIdxBufferData,
                             distanceConstraintsIdxBufferSize));

    cutilSafeCall(cudaMemcpy(pd.d_distanceConstraintsIdxBufferData,
                             g.d_edgeIdxBufferData,
                             distanceConstraintsIdxBufferSize,
                             cudaMemcpyDeviceToDevice));

    size_t edgeIdxBufferSize = g.d_nEdgeIdxBufferElems * sizeof(uint);
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
    // Distance constraint graph
    std::map<std::pair<uint, uint>, std::set<std::pair<uint, uint>>> dcGraph;

    size_t maxEdgeAdjacencySetSize = 0;

    for (uint i = 0; i < g.d_nEdgeIdxBufferElems; i += 2)
    {
        auto currEdge = makeOrderedIdxPair(h_edgeIdxBufferData[i], h_edgeIdxBufferData[i + 1]);

        const auto [currEdgeItr, success] = dcGraph.insert({currEdge, {}});

        for (auto &[edge, adjEdges] : dcGraph)
        {
            auto &[idx1, idx2] = edge;
            if ((currEdge.first == idx1 || currEdge.first == idx2 ||
                 currEdge.second == idx1 || currEdge.second == idx2) &&
                currEdge != edge)
            {
                currEdgeItr->second.insert(edge);
                adjEdges.insert(currEdge);
                auto adjEdgesSize = adjEdges.size();
                if (maxEdgeAdjacencySetSize < adjEdgesSize)
                    maxEdgeAdjacencySetSize = adjEdgesSize;
            }
        }
    }

    std::vector<std::set<std::pair<uint, uint>>> colorSets(maxEdgeAdjacencySetSize + 1);

    for (auto &[edge, adjEdges] : dcGraph)
    {
        for (auto &colorSet : colorSets)
        {
            bool adjEdgeInColorSet = false;
            for (const auto &adjEdge : adjEdges)
            {
                if (auto search = colorSet.find(adjEdge); search != colorSet.end())
                {
                    adjEdgeInColorSet = true;
                    break;
                }
            }
            if (adjEdgeInColorSet)
            {
                continue;
            }
            colorSet.insert(edge);
            break;
        }
    }

    for (size_t i = 0; i < colorSets.size();)
    {
        auto &colorSet = colorSets[i];
        if (colorSet.size() == 0)
        {
            colorSets.erase(colorSets.begin() + i);
        }
        else
        {
            i++;
        }
    }

    pd.d_nDistanceConstraintSets = colorSets.size();

    size_t nDistanceConstraintsPerSetSize = pd.d_nDistanceConstraintSets * sizeof(uint);

    cutilSafeCall(cudaMalloc(&pd.d_nDistanceConstraintsPerSet,
                             nDistanceConstraintsPerSetSize));

    size_t nDistanceConstraintSets = pd.d_nDistanceConstraintSets * sizeof(uint *);

    uint **h_distanceConstraintSets = static_cast<uint **>(malloc(nDistanceConstraintSets));

    cutilSafeCall(cudaMalloc(&pd.d_distanceConstraintSets,
                             nDistanceConstraintSets));

    for (size_t i = 0; i < colorSets.size(); i++)
    {
        auto &colorSet = colorSets[i];
        uint colorSetSize = static_cast<uint>(colorSet.size()) * 2;
        cutilSafeCall(cudaMemcpy(&pd.d_nDistanceConstraintsPerSet[i],
                                 &colorSetSize,
                                 sizeof(uint),
                                 cudaMemcpyHostToDevice));

        std::vector<uint> colorSetVector;
        colorSetVector.reserve(colorSet.size() * 2);
        for (const auto &edge : colorSet)
        {
            colorSetVector.push_back(edge.first);
            colorSetVector.push_back(edge.second);
        }

        size_t distanceConstraintSetSize = colorSetVector.size() * sizeof(uint);
        cutilSafeCall(cudaMalloc(&h_distanceConstraintSets[i],
                                 distanceConstraintSetSize));

        cutilSafeCall(cudaMemcpy(h_distanceConstraintSets[i],
                                 colorSetVector.data(),
                                 distanceConstraintSetSize,
                                 cudaMemcpyHostToDevice));
    }
    cutilSafeCall(cudaMemcpy(pd.d_distanceConstraintSets,
                             h_distanceConstraintSets,
                             nDistanceConstraintSets,
                             cudaMemcpyHostToDevice));

    // Vertex velocities
    cutilSafeCall(cudaMalloc(&pd.d_vertexVelocitiesBufferData,
                             g.d_nVertexPositionBufferElems * sizeof(float)));
    cutilSafeCall(cudaMemset(pd.d_vertexVelocitiesBufferData,
                             0.f,
                             g.d_nVertexPositionBufferElems * sizeof(float)));

    // Vertex masses
    size_t nVertices = g.d_nVertexPositionBufferElems / size_t(3);
    cutilSafeCall(cudaMalloc(&pd.d_vertexMassesBufferData,
                             nVertices * sizeof(float)));

    std::vector<float> vertexMasses(nVertices, 1.f);
    cutilSafeCall(cudaMemcpy(pd.d_vertexMassesBufferData,
                             vertexMasses.data(),
                             nVertices * sizeof(float),
                             cudaMemcpyHostToDevice));
}

__device__ std::pair<uint, uint> getVertexPosIdxRange(uint vPosIdx)
{
    return {vPosIdx * 3, vPosIdx * 3 + 2};
}

__device__ void pbdIteration(PBDGeometry &g, uint nIterations, uint dcIdx1, uint dcIdx2)
{
    auto &pd = g.pbdData;

    auto &vPosBuff = g.d_vertexPositionBufferData;
    auto &distConstraintsIdxBuff = pd.d_distanceConstraintsIdxBufferData;
    auto &vMassesBuff = pd.d_vertexMassesBufferData;
    auto &restLengthBuff = pd.d_distanceConstraintLengthBufferData;

    uint nVertices = static_cast<uint>(g.d_nVertexPositionBufferElems / 3);

    uint vIdx1 = distConstraintsIdxBuff[dcIdx1];
    uint vIdx2 = distConstraintsIdxBuff[dcIdx2];

#ifdef DEBUG
    auto [vPosIdx1Start, vPosIdx1End] = getVertexPosIdxRange(vIdx1);
    auto [vPosIdx2Start, vPosIdx2End] = getVertexPosIdxRange(vIdx2);

    if (!(vPosIdx1End < g.d_nVertexPositionBufferElems) ||
        !(vPosIdx2End < g.d_nVertexPositionBufferElems))
        return;
#else
    auto vPosIdx1Start = vIdx1 * 3;
    auto vPosIdx2Start = vIdx2 * 3;
#endif

    ei::Map<ei::Vector3f> vPos1(&vPosBuff[vPosIdx1Start]);
    ei::Map<ei::Vector3f> vPos2(&vPosBuff[vPosIdx1Start]);

    ei::Vector3f diff = vPos1 - vPos2;
    float dist = diff.norm();

#ifdef DEBUG
    if (!(vIdx1 < nVertices) ||
        !(vIdx1 < nVertices))
        return;
#endif

    float &w1 = vMassesBuff[vIdx1];
    float &w2 = vMassesBuff[vIdx2];

    float invW1 = w1 ? 1.f / w1 : FLT_MAX;
    float invW2 = w2 ? 1.f / w2 : FLT_MAX;

    float W = (invW1 == FLT_MAX || invW2 == FLT_MAX) ? FLT_MAX : invW1 + invW2;

    if (W <= 0 || dist <= 0)
    {
        return;
    }

    float kPrime = 1.f - std::pow(1.0f - 0.5f, 1.f / static_cast<float>(nIterations));

    // TODO figure out how to get the rest length
    float restLength = restLengthBuff[0];

    vPos1 += (-(invW1 / W) *
              (dist - restLength) *
              (diff / dist)) *
             kPrime;

    vPos2 += (-(invW2 / W) *
              (dist - restLength) *
              (diff / dist)) *
             kPrime;
}

__global__ void pbdStep(PBDGeometry g, uint nIterations)
{
    auto &pd = g.pbdData;

    uint threadFlatIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint i; i < pd.d_nDistanceConstraintSets; i++)
    {
        uint startIdx = threadFlatIdx * 2;
        uint endIdx = startIdx + 1;

        if (pd.d_nDistanceConstraintsPerSet[i] < endIdx)
        {
            continue;
        }

        auto &distanceConstraints = pd.d_distanceConstraintSets[i];

        uint dcIdx1 = distanceConstraints[startIdx];
        uint dcIdx2 = distanceConstraints[endIdx];

        for (uint k = 0; k < nIterations; k++)
        {
            pbdIteration(g, nIterations, dcIdx1, dcIdx2);
        }
    }
}

void runPBDSolver(PBDGeometry &g,
                  WorldProperties &worldProperties)
{
    uint nWarps = 1;
    pbdStep<<<1, nWarps>>>(g, 10);
}
