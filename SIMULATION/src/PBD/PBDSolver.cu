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
                                     float *dcRestLengthBufferData)
{
    uint threadFlatIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Get the index range we're operating on (i.e first and second vertex indices)
    uint edgeStartIdx = threadFlatIdx * 2;
    uint edgeEndIdx = edgeStartIdx + 1;

    if (!(edgeEndIdx < nEdgeIdxBufferElems))
    {
        dcRestLengthBufferData[threadFlatIdx] = FLT_MAX;
        return;
    }

    uint vert1StartIdx = edgeIdxBufferData[edgeStartIdx] * 3;
    uint vert2StartIdx = edgeIdxBufferData[edgeEndIdx] * 3;

#ifdef DEBUG
    uint vert1EndIdx = vert1StartIdx + 2;
    uint vert2EndIdx = vert2StartIdx + 2;

    if (!(vert1EndIdx < nVertexPositionBufferElems) ||
        !(vert2EndIdx < nVertexPositionBufferElems))
        return;
#endif

    ei::Map<ei::Vector3f> vert1(&vertexPositionBufferData[vert1StartIdx]);
    ei::Map<ei::Vector3f> vert2(&vertexPositionBufferData[vert2StartIdx]);

    ei::Vector3f diff = vert1 - vert2;

    float restLength = 0;
    if (vert1 == vert2)
        restLength = FLT_MAX;
    else
        restLength = diff.norm();

    dcRestLengthBufferData[threadFlatIdx] = restLength;
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

    // Default init to 0
    uint nIsVertexFixedElems = g.d_nVertexPositionBufferElems / (3 * BYTE_BITS);

    std::vector<byte> isVertexFixed(nIsVertexFixedElems, byte(0));
    for (uint i = 0; i < nFixedVertexIdxsElems; i++)
    {
        uint vertexIdx = fixedVertexIdxs[i];
        setBoolFromPackedBuffer(isVertexFixed.data(), isVertexFixed.size(), vertexIdx, true);
    }
    pd.d_nIsVertexFixedBufferElems = nIsVertexFixedElems;
    // redundant ik but consistent
    size_t isVertexFixedBufferSize = nIsVertexFixedElems * sizeof(byte);
    cutilSafeCall(cudaMalloc(&pd.d_isVertexFixedBuffer,
                             isVertexFixedBufferSize));
    cutilSafeCall(cudaMemcpy(pd.d_isVertexFixedBuffer,
                             &isVertexFixed.data()[0],
                             isVertexFixedBufferSize,
                             cudaMemcpyDefault));

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

    // pd.d_nDcRestLengthBufferElems =
    //     static_cast<uint>(g.d_nEdgeIdxBufferElems / 2);
    //
    // size_t dcRestLengthBufferSize = pd.d_nDcRestLengthBufferElems * sizeof(float);
    // cudaMalloc(&pd.d_dcRestLengthBufferData,
    //           dcRestLengthBufferSize);
    //
    // uint nEdges = static_cast<uint>(g.d_nEdgeIdxBufferElems / 2);
    //
    // calculateEdgeLengths<<<1, nEdges>>>(g.d_vertexPositionBufferData,
    //                                    g.d_nVertexPositionBufferElems,
    //                                    g.d_edgeIdxBufferData,
    //                                    g.d_nEdgeIdxBufferElems,
    //                                    pd.d_dcRestLengthBufferData);

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

    size_t nDistanceConstraintSetsSize = pd.d_nDistanceConstraintSets * sizeof(uint);

    cutilSafeCall(cudaMalloc(&pd.d_nDistanceConstraintsPerSet,
                             nDistanceConstraintSetsSize));
    cutilSafeCall(cudaMalloc(&pd.d_nDcRestLengthPerSet,
                             nDistanceConstraintSetsSize));

    size_t distanceConstraintSetsSize = pd.d_nDistanceConstraintSets * sizeof(uint *);
    uint **h_distanceConstraintSets = static_cast<uint **>(malloc(distanceConstraintSetsSize));

    size_t dcRestLengthSetsSize = pd.d_nDistanceConstraintSets * sizeof(float *);
    float **h_dcRestLengthSets = static_cast<float **>(malloc(dcRestLengthSetsSize));

    cutilSafeCall(cudaMalloc(&pd.d_distanceConstraintSets,
                             distanceConstraintSetsSize));
    cutilSafeCall(cudaMalloc(&pd.d_dcRestLengthSets,
                             dcRestLengthSetsSize));

    for (size_t i = 0; i < colorSets.size(); i++)
    {
        auto &colorSet = colorSets[i];

        uint nColorSetElems = static_cast<uint>(colorSet.size()) * 2;
        cutilSafeCall(cudaMemcpy(&pd.d_nDistanceConstraintsPerSet[i],
                                 &nColorSetElems,
                                 sizeof(uint),
                                 cudaMemcpyHostToDevice));

        uint nRestLengthSetElems = static_cast<uint>(colorSet.size());
        cutilSafeCall(cudaMemcpy(&pd.d_nDcRestLengthPerSet[i],
                                 &nRestLengthSetElems,
                                 sizeof(uint),
                                 cudaMemcpyHostToDevice));

        std::vector<uint> colorSetVector;
        colorSetVector.reserve(nColorSetElems);
        for (const auto &edge : colorSet)
        {
            colorSetVector.push_back(edge.first);
            colorSetVector.push_back(edge.second);
        }

        uint nDistanceContraintSetElems = colorSetVector.size();
        size_t distanceConstraintSetSize = nDistanceContraintSetElems * sizeof(uint);
        cutilSafeCall(cudaMalloc(&h_distanceConstraintSets[i],
                                 distanceConstraintSetSize));
        cutilSafeCall(cudaMemcpy(h_distanceConstraintSets[i],
                                 colorSetVector.data(),
                                 distanceConstraintSetSize,
                                 cudaMemcpyHostToDevice));

        size_t dcRestLengthSetSize = colorSet.size() * sizeof(float);
        cutilSafeCall(cudaMalloc(&h_dcRestLengthSets[i],
                                 dcRestLengthSetSize));

        calculateEdgeLengths<<<1, nRestLengthSetElems>>>(g.d_vertexPositionBufferData,
                                                         g.d_nVertexPositionBufferElems,
                                                         h_distanceConstraintSets[i],
                                                         nDistanceContraintSetElems,
                                                         h_dcRestLengthSets[i]);

        if (nDistanceContraintSetElems > pd.d_maxNDistanceConstraintsInSet)
            pd.d_maxNDistanceConstraintsInSet = nDistanceContraintSetElems;
    }
    cutilSafeCall(cudaMemcpy(pd.d_distanceConstraintSets,
                             h_distanceConstraintSets,
                             distanceConstraintSetsSize,
                             cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(pd.d_dcRestLengthSets,
                             h_dcRestLengthSets,
                             dcRestLengthSetsSize,
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

__device__ void pbdIteration(PBDGeometry &g, uint nIterations, uint vIdx1, uint vIdx2, float dcRestLength)
{
    auto &pd = g.pbdData;

    auto &vPosBuff = g.d_vertexPositionBufferData;
    auto &distConstraintsIdxBuff = pd.d_distanceConstraintsIdxBufferData;
    auto &vMassesBuff = pd.d_vertexMassesBufferData;
    float **&dcRestLengthSets = pd.d_dcRestLengthSets;

    uint nVertices = static_cast<uint>(g.d_nVertexPositionBufferElems / 3);

    if (!(vIdx1 < nVertices) ||
        !(vIdx2 < nVertices))
    {
        printf("Index out of range:\nvIdx1: %i\nvIdx2: %i\n", vIdx1, vIdx2);
        return;
    }

    bool isV1Fixed = getBoolFromPackedBuffer(pd.d_isVertexFixedBuffer,
                                             pd.d_nIsVertexFixedBufferElems,
                                             vIdx1);
    bool isV2Fixed = getBoolFromPackedBuffer(pd.d_isVertexFixedBuffer,
                                             pd.d_nIsVertexFixedBufferElems,
                                             vIdx2);

    if (isV1Fixed && isV2Fixed)
    {
        printf("Both vertices are fixed: vIdx1=%i, vIdx2=%i\n",
               vIdx1, vIdx2);
        return;
    }

    auto [vPosIdx1Start, vPosIdx1End] = getVertexPosIdxRange(vIdx1);
    auto [vPosIdx2Start, vPosIdx2End] = getVertexPosIdxRange(vIdx2);

    if (!(vPosIdx1End < g.d_nVertexPositionBufferElems) ||
        !(vPosIdx2End < g.d_nVertexPositionBufferElems))
    {
        printf("Index out of range:\nvPosIdx1End: %i\nvPosIdx2End: %i\n", vPosIdx1End, vPosIdx2End);
        return;
    }

    ei::Map<ei::Vector3f> vPos1(&vPosBuff[vPosIdx1Start]);
    ei::Map<ei::Vector3f> vPos2(&vPosBuff[vPosIdx2Start]);

    ei::Vector3f diff = vPos1 - vPos2;
    float dist = diff.norm();

    float &w1 = vMassesBuff[vIdx1];
    float &w2 = vMassesBuff[vIdx2];

    float invW1 = w1 != 0 ? (isV1Fixed ? 0 : 1.f / w1) : FLT_MAX;
    float invW2 = w2 != 0 ? (isV2Fixed ? 0 : 1.f / w2) : FLT_MAX;

    float W = (invW1 == FLT_MAX || invW2 == FLT_MAX) ? FLT_MAX : invW1 + invW2;

    // printf("W %f, dist %f\n", W, dist);
    if (W <= 0 || dist <= 0)
    {
        return;
    }

    float kPrime = 1.f - std::pow(0.5f, 1.f / static_cast<float>(nIterations));
    // printf("Setting pos for vertex at idx: %i\n", vIdx1);
    vPos1 += -(invW1 / W) *
             (dist - dcRestLength) *
             (diff.normalized()) *
             kPrime;

    // printf("Setting pos for vertex at idx: %i\n", vIdx2);
    vPos2 += (invW2 / W) *
             (dist - dcRestLength) *
             (diff.normalized()) *
             kPrime;
}

__global__ void pbdStep(PBDGeometry g, uint nIterations)
{
    auto &pd = g.pbdData;

    uint threadFlatIdx = blockIdx.x * blockDim.x + threadIdx.x;

    uint startIdx = threadFlatIdx * 2;
    uint endIdx = startIdx + 1;

    // printf("n Distance constraint sets, %i for thread %i\n", pd.d_nDistanceConstraintSets, threadFlatIdx);
    for (uint i = 0; i < pd.d_nDistanceConstraintSets; i++)
    {
#ifdef DEBUG
        printf("in debug mode");
#endif

        // printf("N distance constraints in set %i\n", pd.d_nDistanceConstraintsPerSet[i]);
        if (pd.d_nDistanceConstraintsPerSet[i] < endIdx)
        {
            // printf("continuing for thread %i\n", threadFlatIdx);
            continue;
        }

        auto &distanceConstraintSet = pd.d_distanceConstraintSets[i];
        auto &dcRestLengths = pd.d_dcRestLengthSets[i];

        uint vIdx1 = distanceConstraintSet[startIdx];
        uint vIdx2 = distanceConstraintSet[endIdx];

        float dcRestLength = dcRestLengths[threadFlatIdx];

        for (uint k = 0; k < nIterations; k++)
        {
            pbdIteration(g, nIterations, vIdx1, vIdx2, dcRestLength);
        }
    }
}

// void runPBDSolver(PBDGeometry &g,
//                   WorldProperties &worldProperties)
void runPBDSolver(PBDGeometry &g)
{
    auto &pd = g.pbdData;
    uint nWarps = static_cast<uint>(pd.d_maxNDistanceConstraintsInSet / 2);
    pbdStep<<<1, nWarps>>>(g, 10);
}

__global__ void applyGravity(PBDGeometry g, WorldProperties wp)
{
    uint threadFlatIdx = blockIdx.x * blockDim.x + threadIdx.x;

    uint vertexIdx = threadFlatIdx;

    bool isFixedVerted = getBoolFromPackedBuffer(g.pbdData.d_isVertexFixedBuffer, g.d_nVertexPositionBufferElems / 3, vertexIdx);

    if (isFixedVerted)
    {
        printf("applyGravity - vertex index out of range: %i\n", vertexIdx);
        return;
    }

    uint startIdx = threadFlatIdx * 3;
    uint endIdx = startIdx + 2;

    if (!(endIdx < g.d_nVertexPositionBufferElems))
    {
        printf("applyGravity - vertex index out of range: %i\n", startIdx);
        return;
    }

    ei::Map<ei::Vector3f> vertex(&g.d_vertexPositionBufferData[startIdx]);
    vertex += wp.m_gravConstant * wp.m_timeStep;
}

void applyExternalForces(PBDGeometry &g, WorldProperties &wp)
{
    applyGravity<<<1, (g.d_nVertexPositionBufferElems / 3)>>>(g, wp);
}
