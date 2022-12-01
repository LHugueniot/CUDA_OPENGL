#include "gtest/gtest.h"

#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "Geometry/Geometry.cuh"
#include "Geometry/LoadGeometry.cuh"
#include "Sim/PBD/PBDSolver.cuh"
#include "Viewer/Camera.h"

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STR(s) #s
#define XSTR(s) STR(s)

static constexpr char *kAssetDirectory = XSTR(ASSETS_DIRECTORY);

template <typename T, typename P = float>
bool _EXPECT_NEAR(T a, T b, P p)
{
    EXPECT_NEAR(a, b, p);
    T diff = std::abs(a - b);
    return p > diff && diff >= 0;
}

template <typename T, typename P = float>
void EXPECT_NEAR_STDVEC(T a, T b, P p = 0.00001)
{
    ASSERT_EQ(a.size(), b.size());
    bool val = true;
    for (int i = 0; i < a.size(); i++)
    {
        val &= _EXPECT_NEAR(a.data()[i], b.data()[i], p);
    }
    if (!val)
    {
        EXPECT_EQ(a, b);
    }
}

template <typename T, typename P = float>
void EXPECT_NEAR_MATRIX(T a, T b, P p = 0.00001)
{
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    bool val = true;
    for (int i = 0; i < a.rows() * a.cols(); i++)
    {
        val &= _EXPECT_NEAR(a.data()[i], b.data()[i], p);
    }
    if (!val)
    {
        EXPECT_EQ(a, b);
    }
}

void print_glm_mat4(glm::mat4 &mat)
{
    for (int r = 0; r < 4; r++)
    {
        for (int c = 0; c < 4; c++)
        {
            fprintf(stdout, "%7.1f", mat[r][c]);
        }
        fprintf(stdout, "\n");
    }
}

template <typename T>
std::vector<T> deviceToContainer(T *d_ptr, size_t nElems)
{
    std::vector<T> result(nElems);

    assert(d_ptr != nullptr);
    assert(result.data() != nullptr);

    cutilSafeCall(cudaMemcpy(result.data(),
                             d_ptr,
                             nElems * sizeof(T),
                             cudaMemcpyDeviceToHost));
    return result;
}

float length(ei::Vector3f const &v)
{
    return sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
}

TEST(EigenTests, ProjectionAndViewTest)
{
    float radians_fov = 45.0f;
    float windowWidth = 1920.f;
    float windowHeight = 1080.f;
    float fov = 45.f;
    float far = 100.0f;
    float near = 0.1f;

    glm::mat4 glm_projection = glm::perspective(
        glm::radians(radians_fov), windowWidth / windowHeight, near, far);
    glm::mat4 glm_view = glm::lookAt(
        glm::vec3(0, 0, 30), // Camera is at (4,3,-3), in World Space
        glm::vec3(0, 0, 0),  // and looks at the origin
        glm::vec3(0, 1, 0)   // Head is up (set to 0,-1,0 to look upside-down)
    );

    glm::mat4 glm_PV = glm_projection * glm_view;

    ei::Matrix4f ei_projection;
    ei::Matrix4f ei_view;

    ei_utils::setProjMat(ei_projection, windowWidth, windowHeight, TO_RAD(fov), far, near);
    ei_utils::setLookAt(ei_view, {0, 0, 30}, {0, 0, 0}, {0, 1, 0});

    // ei::Matrix4f ei_PV = ei_projection.transpose() * ei_view.transpose();
    ei::Matrix4f ei_PV = ei_projection * ei_view;

    ASSERT_EQ(TO_RAD(fov), glm::radians(radians_fov));

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(glm::value_ptr(glm_projection)[i], ei_projection.data()[i], 0.1);
        EXPECT_NEAR(glm::value_ptr(glm_view)[i], ei_view.data()[i], 0.1);
        EXPECT_NEAR(glm::value_ptr(glm_PV)[i], ei_PV.data()[i], 0.1);
    }

    //    std::cout << "PROJECTION MATRICES" << std::endl;
    //    std::cout << ei_projection << std::endl;
    //    print_glm_mat4(glm_projection);
    //
    //    std::cout << "VIEW MATRICES" << std::endl;
    //    std::cout << ei_view << std::endl;
    //    print_glm_mat4(glm_view);
    //
    //    std::cout << "PROJECTION VIEW MATRICES" << std::endl;
    //    std::cout << ei_PV.transpose() << std::endl;
    //    print_glm_mat4(glm_PV);
    //    //std::cout << glm::to_string(glm_projection) << std::endl;
    //
    //
    //    for (int i=0; i<16; i++)
    //    {
    //        auto* p = &glm_projection[0][0];
    //        assert(p[i] == ei_projection.data()[i]);
    //        auto* v = &glm_view[0][0];
    //        assert(v[i] == ei_view.data()[i]);
    //        auto* pv = &glm_PV[0][0];
    //        assert(pv[i] == ei_PV.data()[i]);
    //    }
    //
    //    for (int r=0; r<4; r++)
    //        for (int c=0; c<4; c++)
    //        {
    //            assert(glm_projection[r][c] == ei_projection(r, c));
    //            assert(glm_view[r][c] == ei_view(r, c));
    //            assert(glm_PV[r][c] == ei_PV(r, c));
    //        }
}

using MapVector3f = Eigen::Map<ei::Vector3f>;

void setupTestGeom(Geometry &testGeom,
                   std::vector<float> &vertexData)
{
    testGeom.d_nVertexPositionBufferElems = vertexData.size();

    size_t bufferBytesSize = testGeom.d_nVertexPositionBufferElems *
                             sizeof(float);

    cudaMalloc((void **)&testGeom.d_vertexPositionBufferData,
               bufferBytesSize);

    cudaMemcpy(testGeom.d_vertexPositionBufferData,
               &vertexData.data()[0],
               bufferBytesSize,
               cudaMemcpyHostToDevice);
}

void retrieveProcessedGeom(Geometry &testGeom,
                           std::vector<float> &vertexData)
{
    vertexData.resize(testGeom.d_nVertexPositionBufferElems);
    size_t bufferBytesSize = testGeom.d_nVertexPositionBufferElems *
                             sizeof(float);
    cudaMemcpy(static_cast<void *>(&vertexData.data()[0]),
               static_cast<const void *>(testGeom.d_vertexPositionBufferData),
               bufferBytesSize,
               cudaMemcpyDeviceToHost);

    cudaFree(testGeom.d_vertexPositionBufferData);
}

TEST(Geometry, scaleGeom1)
{
    auto testGeom = Geometry();
    std::vector<float> vertexData =
        {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1};

    ei::Vector3f scale = {2.5f, 5.0f, 10.0f};
    setupTestGeom(testGeom, vertexData);

    scaleGeom(testGeom, scale);

    retrieveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
        {
            0, 0, 0,
            2.5f, 0, 0,
            0, 5.0f, 0,
            0, 0, 10.0f};

    ASSERT_EQ(vertexData, expectedVertexData);
}

TEST(Geometry, scaleGeom2)
{
    auto testGeom = Geometry();
    std::vector<float> vertexData =
        {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1};

    ei::Vector3f scale = {5.f, 5.f, 5.f};
    setupTestGeom(testGeom, vertexData);

    scaleGeom(testGeom, scale);

    retrieveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
        {
            0, 0, 0,
            5.f, 0, 0,
            0, 5.f, 0,
            0, 0, 5.f};

    ASSERT_EQ(vertexData, expectedVertexData);
}

TEST(Geometry, scaleGeom3)
{
    auto testGeom = Geometry();
    std::vector<float> vertexData =
        {
            0, 0, 0,
            10.f, 0, 0,
            0, 10.f, 0,
            0, 0, 10.f};

    ei::Vector3f scale = {2.f, 2.f, 2.f};
    ei::Vector3f pivot = {5.f, 5.f, 5.f};
    setupTestGeom(testGeom, vertexData);

    scaleGeom(testGeom, scale, pivot);

    retrieveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
        {
            -5, -5, -5,
            15, -5, -5,
            -5, 15, -5,
            -5, -5, 15};

    ASSERT_EQ(vertexData, expectedVertexData);
}

TEST(Geometry, rotateGeom1)
{
    auto testGeom = Geometry();
    std::vector<float> vertexData =
        {
            0, 0, 0,
            10.f, 0, 0,
            0, 10.f, 0,
            0, 0, 10.f};

    ei::Vector3f axis = {0, 1.f, 0};

    setupTestGeom(testGeom, vertexData);

    rotateGeom(testGeom, axis, TO_RAD(90), {0, 0, 0});

    retrieveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
        {
            0, 0, 0,
            0, 0, 10.f,
            0, 10.f, 0,
            -10.f, 0, 0};

    EXPECT_NEAR_STDVEC(vertexData, expectedVertexData);
}

TEST(Geometry, rotateGeom2)
{
    auto testGeom = Geometry();
    std::vector<float> vertexData =
        {
            0, 0, 0,
            10.f, 0, 0,
            0, 10.f, 0,
            0, 0, 10.f};

    ei::Vector3f axis = {0, 1.f, 0};

    setupTestGeom(testGeom, vertexData);

    rotateGeom(testGeom, axis, TO_RAD(90), {5.f, 0, 0});

    retrieveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
        {
            5.f, 0, -5.f,
            5.f, 0, 5.f,
            5.f, 10.f, -5.f,
            -5.f, 0, -5.f};

    EXPECT_NEAR_STDVEC(vertexData, expectedVertexData);
}

TEST(Geometry, translateGeom1)
{
    auto testGeom = Geometry();
    std::vector<float> vertexData =
        {
            0, 0, 0,
            10.f, 0, 0,
            0, 10.f, 0,
            0, 0, 10.f};

    ei::Vector3f translation = {-10.f, -10.f, -10.f};
    setupTestGeom(testGeom, vertexData);

    translateGeom(testGeom, translation);

    retrieveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
        {
            -10.f, -10.f, -10.f,
            0.f, -10.f, -10.f,
            -10.f, 0.f, -10.f,
            -10.f, -10.f, 0.f};

    EXPECT_NEAR_STDVEC(vertexData, expectedVertexData);
}

TEST(Geometry, transformGeom1)
{
    auto testGeom = Geometry();
    std::vector<float> vertexData =
        {
            1, 1, 1};

    ei::Transform3f transform = ei::Translation3f({10.f, 10.f, 10.f}) *
                                ei::AngleAxisf(TO_RAD(90), ei::Vector3f{0.f, 1.f, 0.f}) *
                                ei::Scaling(10.f, 10.f, 10.f);

    setupTestGeom(testGeom, vertexData);

    transformGeom(testGeom, transform);

    retrieveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
        {
            -110.f, 110.f, -110.f};

    EXPECT_NEAR_STDVEC(vertexData, expectedVertexData, 0.001);
}

TEST(Geometry, LoadGeometry)
{
    const aiScene *sceneCache = nullptr;

    std::filesystem::path assetDir = std::filesystem::absolute(kAssetDirectory);
    std::filesystem::path assetFile = assetDir / "cube_simple.obj";

    std::vector<const aiMesh *> meshes = loadAiMeshes(assetFile, &sceneCache);

    ASSERT_EQ(meshes.size(), 1);

    struct MeshData
    {
        std::vector<float> m_vertexData;
        std::vector<uint> m_edgeIdxs;
        std::vector<uint> m_faceIdxs;
    };
    std::map<std::string, MeshData> nameToMeshData;

    for (auto meshPtr : meshes)
    {
        MeshData &meshData = nameToMeshData[meshPtr->mName.data] = MeshData();

        meshData.m_vertexData.reserve(meshPtr->mNumVertices * 3);

        // std::cout << meshPtr->mNumVertices << std::endl;
        for (uint v = 0; v < meshPtr->mNumVertices; v++)
        {
            for (uint d = 0; d < 3; d++)
            {
                meshData.m_vertexData.push_back(meshPtr->mVertices[v][d]);
            }
        }

        // Assume faces have only 3 vertices
        meshData.m_faceIdxs.reserve(meshPtr->mNumFaces * 3);

        for (uint f = 0; f < meshPtr->mNumFaces; ++f)
        {
            const aiFace &face = meshPtr->mFaces[f];

            for (uint fi = 0; fi < face.mNumIndices; fi++)
            {
                meshData.m_faceIdxs.push_back(face.mIndices[fi]);
            }
        }
    }
    ASSERT_EQ(nameToMeshData.size(), 1);

    ASSERT_NE(nameToMeshData.find("Cube"), nameToMeshData.end());

    auto &meshData = nameToMeshData["Cube"];

    // printStdVecInStride(meshData.m_faceIdxs);

    ASSERT_EQ(meshData.m_vertexData.size(), 24);
    ASSERT_EQ(meshData.m_faceIdxs.size(), 36);

    std::vector<std::pair<std::string, Geometry *>> nameToGeometry =
        initGeometryFromAiMeshes<Geometry>(meshes);

    // std::vector<std::pair<std::string, Geometry *>> nameToGeometry =
    //     initGeometryFromAiMeshes<Geometry,
    //                              CuGlBufferSetter<float>,
    //                              CuGlBufferSetter<uint>,
    //                              CuGlBufferSetter<uint>>(meshes);

    Geometry *gPtr;
    for (auto &[name, geomPtr] : nameToGeometry)
        if (name == "Cube")
            gPtr = geomPtr;

    ASSERT_NE(gPtr, nullptr);

    Geometry &g = *gPtr;

    std::vector<float> vertexBuffer = deviceToContainer(
        g.d_vertexPositionBufferData,
        g.d_nVertexPositionBufferElems);

    ASSERT_EQ(vertexBuffer.size(), 8 * 3);

    std::vector<uint> edgeIndices = deviceToContainer(
        g.d_edgeIdxBufferData,
        g.d_nEdgeIdxBufferElems);

    ASSERT_EQ(edgeIndices.size(), 18 * 2);

    for (auto [name, geometry] : nameToGeometry)
    {
        std::vector<float> vertexData;
        retrieveProcessedGeom(*geometry, vertexData);

        ASSERT_NE(nameToMeshData.find(name), nameToMeshData.end());

        ASSERT_EQ(vertexData, nameToMeshData[name].m_vertexData);
    }

    aiReleaseImport(sceneCache);
}

using NameToPBDGeomPtr = std::vector<std::pair<std::string, PBDGeometry *>>;

std::pair<PBDGeometry *, NameToPBDGeomPtr> pbdGeomSetup()
{
    const aiScene *sceneCache = nullptr;

    std::filesystem::path assetDir = std::filesystem::absolute(kAssetDirectory);
    std::filesystem::path assetFile = assetDir / "cube_simple.obj";

    std::vector<const aiMesh *> meshes = loadAiMeshes(assetFile, &sceneCache);

    NameToPBDGeomPtr nameToGeometry = initGeometryFromAiMeshes<PBDGeometry>(meshes);

    // std::cout << edgeSetter.m_data << std::endl;

    PBDGeometry *cubeGeom = nullptr;
    for (auto &[name, geomPtr] : nameToGeometry)
        if (name == "Cube")
            cubeGeom = geomPtr;

    return {cubeGeom, nameToGeometry};
}

TEST(PBDGeometry, initializePBDParameters)
{
    uint fixedVertexIdx = 0;
    uint nFixedVertexIdx = 1;
    auto [gPtr, nameToGeometry] = pbdGeomSetup();

    ASSERT_NE(gPtr, nullptr);
    auto &g = *gPtr;

    auto &p = g.pbdData;

    initializePBDParameters(g, &fixedVertexIdx, nFixedVertexIdx);

    std::vector<byte> isVertexFixedBuffer = deviceToContainer(
        p.d_isVertexFixedBuffer,
        p.d_nIsVertexFixedBufferElems);

    ASSERT_EQ(p.d_nIsVertexFixedBufferElems, 1);

    std::vector<byte> expectedIsVertexFixedBuffer{
        byte(1)};

    ASSERT_EQ(isVertexFixedBuffer.size(), 1);

    ASSERT_EQ(isVertexFixedBuffer, expectedIsVertexFixedBuffer);

    std::vector<uint> fixedVertexIdxBufferData = deviceToContainer(
        p.d_fixedVertexIdxBufferData,
        p.d_nFixedVertexIdxBufferElems);

    ASSERT_EQ(p.d_nFixedVertexIdxBufferElems, 1);
    ASSERT_EQ(fixedVertexIdxBufferData.size(), 1);

    std::vector<float> vertexPositions = deviceToContainer(
        g.d_vertexPositionBufferData,
        g.d_nVertexPositionBufferElems);

    std::vector<uint> edgeIndices = deviceToContainer(
        g.d_edgeIdxBufferData,
        g.d_nEdgeIdxBufferElems);

    auto &getVector3fs = [](std::vector<float> v, size_t i)
    {
        return ei::Vector3f{{v[i], v[i + 1], v[i + 2]}};
    };
    auto &get2Vector3fs = [](std::vector<float> v, size_t i)
    {
        return std::pair<ei::Vector3f, ei::Vector3f>{
            {v[i], v[i + 1], v[i + 2]},
            {v[i + 3], v[i + 4], v[i + 5]}};
    };

    ASSERT_EQ(edgeIndices.size(), 18 * 2);
    ASSERT_EQ(vertexPositions.size() % 6, 0);

    std::vector<uint> distanceConstraintsIndices = deviceToContainer(
        p.d_distanceConstraintsIdxBufferData,
        p.d_nDistanceConstraintsIdxBufferElems);

    ASSERT_EQ(distanceConstraintsIndices, edgeIndices);

    // Check max num distance constraints
    ASSERT_EQ(p.d_maxNDistanceConstraintsInSet, 8);

    // Distance constraint sets
    std::vector<std::vector<uint>> distanceConstraintSets;
    std::vector<uint *> distanceConstraintSetsDevicePtrs = deviceToContainer(
        p.d_distanceConstraintSets,
        p.d_nDistanceConstraintSets);
    std::vector<uint> nDistanceConstraintsPerSet = deviceToContainer(
        p.d_nDistanceConstraintsPerSet,
        p.d_nDistanceConstraintSets);

    // Dc rest length
    std::vector<std::vector<float>> dcRestLengthSets;
    std::vector<float *> dcRestLengthsSetsDevicePtrs = deviceToContainer(
        p.d_dcRestLengthSets,
        p.d_nDistanceConstraintSets);
    std::vector<uint> nDcRestLengthPerSet = deviceToContainer(
        p.d_nDcRestLengthPerSet,
        p.d_nDistanceConstraintSets);

    for (size_t i = 0; i < p.d_nDistanceConstraintSets; i++)
    {
        auto nDistanceConstraints = nDistanceConstraintsPerSet[i];
        distanceConstraintSets.push_back(
            deviceToContainer(
                distanceConstraintSetsDevicePtrs[i],
                nDistanceConstraints));

        auto nDcRestLengths = nDcRestLengthPerSet[i];
        ASSERT_EQ(nDcRestLengths, nDistanceConstraints / 2);

        dcRestLengthSets.push_back(
            deviceToContainer(
                dcRestLengthsSetsDevicePtrs[i],
                nDcRestLengths));
    }

    std::cout << distanceConstraintSets << std::endl;

    // Assert there is maximum one reference to a vertex index in each set
    for (const auto &dcSet : distanceConstraintSets)
    {
        std::map<uint, uint> idxToFrequency;

        for (auto &&idx : dcSet)
        {
            const auto [currEdgeItr, success] = idxToFrequency.insert({idx, 0});

            idxToFrequency[idx] += 1;
            ASSERT_EQ(currEdgeItr->second, 1);
        }
    }

    std::vector<float> dcRestLengths;

    for (const auto &dcRestLengthSet : dcRestLengthSets)
    {
        for (auto &&restLength : dcRestLengthSet)
        {
            dcRestLengths.push_back(restLength);
        }
    }

    std::vector<float> expectedDcRestLengths;

    for (const auto &dcSet : distanceConstraintSets)
    {
        for (size_t i = 0; i < dcSet.size(); i += 2)
        {
            uint vIdx1 = dcSet[i] * 3;
            uint vIdx2 = dcSet[i + 1] * 3;

            ei::Map<ei::Vector3f> vPos1(&vertexPositions[vIdx1]);
            ei::Map<ei::Vector3f> vPos2(&vertexPositions[vIdx2]);
            ei::Vector3f diff = vPos1 - vPos2;
            float restLength = diff.norm();
            expectedDcRestLengths.push_back(restLength);
        }
    }
    ASSERT_EQ(dcRestLengths, expectedDcRestLengths);

    std::vector<float> vertexVelocitiesBuffer = deviceToContainer(
        p.d_vertexVelocitiesBufferData,
        g.d_nVertexPositionBufferElems);

    ASSERT_EQ(vertexVelocitiesBuffer.size(), 24);
    for (auto &&e : vertexVelocitiesBuffer)
    {
        ASSERT_EQ(e, 0.f);
    }

    size_t nVertices = g.d_nVertexPositionBufferElems / size_t(3);
    std::vector<float> vertexMassesBuffer = deviceToContainer(
        p.d_vertexMassesBufferData,
        nVertices);

    ASSERT_EQ(vertexMassesBuffer.size(), 8);
    for (auto &&e : vertexMassesBuffer)
    {
        ASSERT_EQ(e, 1.f);
    }
}

TEST(PBDGeometry, runPBDSolver_noExternalForces)
{
    auto [gPtr, nameToGeometry] = pbdGeomSetup();

    ASSERT_NE(gPtr, nullptr);
    auto &g = *gPtr;
    auto &p = g.pbdData;

    uint fixedVertexIdx = 0;
    uint nFixedVertexIdx = 1;
    initializePBDParameters(g, &fixedVertexIdx, nFixedVertexIdx);

    std::vector<float> preSolverVertexPositions = deviceToContainer(
        g.d_vertexPositionBufferData,
        g.d_nVertexPositionBufferElems);

    runPBDSolver(g);

    std::vector<float> postSolverVertexPositions = deviceToContainer(
        g.d_vertexPositionBufferData,
        g.d_nVertexPositionBufferElems);

    // Assert without any exterior forces that vertex positions haven't changed
    ASSERT_EQ(preSolverVertexPositions, postSolverVertexPositions);
}

TEST(PBDGeometry, applyExternalForces)
{
    auto [gPtr, nameToGeometry] = pbdGeomSetup();

    ASSERT_NE(gPtr, nullptr);
    auto &g = *gPtr;
    auto &p = g.pbdData;

    uint fixedVertexIdx = 0;
    uint nFixedVertexIdx = 1;
    initializePBDParameters(g, &fixedVertexIdx, nFixedVertexIdx);

    std::vector<float> expectedSolverVertexPositions = deviceToContainer(
        g.d_vertexPositionBufferData,
        g.d_nVertexPositionBufferElems);

    WorldProperties wp;
    // Start from the second vertex as we expect it to be fixed (fixedVertexIdx)
    for (size_t i = 4; i < expectedSolverVertexPositions.size(); i += 3)
    {
        expectedSolverVertexPositions[i] += wp.m_gravConstant[1] * wp.m_timeStep;
    }
    applyExternalForces(g, wp);

    ASSERT_NE(g.d_vertexPositionBufferData, nullptr);
    // ei::Vector3f translation = {-10.f, -10.f, -10.f};
    // translateGeom(g, translation);

    std::vector<float> postExternalForcesVertexPositions(g.d_nVertexPositionBufferElems);

    ASSERT_NE(g.d_vertexPositionBufferData, nullptr);
    ASSERT_NE(postExternalForcesVertexPositions.data(), nullptr);

    cutilSafeCall(cudaMemcpy(postExternalForcesVertexPositions.data(),
                             g.d_vertexPositionBufferData,
                             g.d_nVertexPositionBufferElems * sizeof(float),
                             cudaMemcpyDeviceToHost));

    // std::vector<float> postExternalForcesVertexPositions = deviceToContainer(
    //     g.d_vertexPositionBufferData,
    //     g.d_nVertexPositionBufferElems);

    ASSERT_EQ(postExternalForcesVertexPositions[1], -1);
    // Assert without any exterior forces that vertex positions haven't changed
    ASSERT_EQ(expectedSolverVertexPositions, postExternalForcesVertexPositions);
}

//__global__ void testPrint()
//{
//    uint threadFlatIdx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    printf("Printing from thread: %i\n", threadFlatIdx);
//}
//
// TEST(InheritanceTest, childMemberAccess)
//{
//    struct A
//    {
//        int a = 0;
//    };
//
//    struct B : A
//    {
//    };
//    const auto childMemberAccessA = [](A &a)
//    {
//        ASSERT_EQ(a.a, 0);
//    };
//    const auto childMemberAccessB = [](B &b)
//    {
//        ASSERT_EQ(b.a, 1);
//    };
//    B b;
//    b.a = 1;
//
//    childMemberAccessB(b);
//    childMemberAccessA(b);
//}
//
// TEST(TESTS_main_cu, testPrint)
//{
//    // testPrint<<<1, 100>>>();
//}

TEST(Sim, PackedBufferBools)
{
    std::vector<byte> boolBuffer(2, byte(0));
    for (uint i = 0; i < BYTE_BITS * 2; i++)
    {
        ASSERT_EQ(getBoolFromPackedBuffer(boolBuffer.data(), boolBuffer.size(), i), false);
    }

    std::array<uint, 4> indices{0, 8, 12, 15};
    for (auto idx : indices)
    {
        setBoolFromPackedBuffer(boolBuffer.data(), boolBuffer.size(), idx, true);
    }
    for (auto idx : indices)
    {
        ASSERT_EQ(getBoolFromPackedBuffer(boolBuffer.data(), boolBuffer.size(), idx), true);
    }

    // for (const auto &e : boolBuffer)
    //{
    //
    //    std::cout << std::bitset<8>(static_cast<char>(e)) << ", ";
    //}
    // std::cout << std::endl;

    for (uint i = 0; i < BYTE_BITS * 2; i++)
    {
        // Skip all the ones we've set
        bool skip = false;
        for (auto idx : indices)
            if (i == idx)
                skip = true;
        if (skip)
            continue;
        ASSERT_EQ(getBoolFromPackedBuffer(boolBuffer.data(), boolBuffer.size(), i), false);
    }
}

TEST(PBDGeometry, runPBDSolver_withExternalForces)
{
    auto [gPtr, nameToGeometry] = pbdGeomSetup();

    ASSERT_NE(gPtr, nullptr);
    auto &g = *gPtr;
    auto &p = g.pbdData;
    uint fixedVertexIdx = 0;
    uint nFixedVertexIdx = 1;

    initializePBDParameters(g, &fixedVertexIdx, nFixedVertexIdx);

    std::vector<byte> isVertexFixedBuffer = deviceToContainer(
        p.d_isVertexFixedBuffer,
        p.d_nIsVertexFixedBufferElems);

    // for (const auto &e : isVertexFixedBuffer)
    //{
    //
    //    std::cout << std::bitset<8>(static_cast<char>(e)) << ", ";
    //}
    // std::cout << std::endl;

    ASSERT_EQ(p.d_nIsVertexFixedBufferElems, 1);

    std::vector<byte> expectedIsVertexFixedBuffer{
        byte(1)};

    ASSERT_EQ(isVertexFixedBuffer.size(), 1);

    ASSERT_EQ(isVertexFixedBuffer, expectedIsVertexFixedBuffer);

    std::vector<float> preSolverVertexPositions = deviceToContainer(
        g.d_vertexPositionBufferData,
        g.d_nVertexPositionBufferElems);

    WorldProperties props;
    applyExternalForces(g, props);

    std::vector<float> postExternalForcesVertexPositions = deviceToContainer(
        g.d_vertexPositionBufferData,
        g.d_nVertexPositionBufferElems);

    runPBDSolver(g);

    std::vector<float> postSolverVertexPositions = deviceToContainer(
        g.d_vertexPositionBufferData,
        g.d_nVertexPositionBufferElems);

    // Assert without any exterior forces that vertex positions haven't changed
    ASSERT_NE(preSolverVertexPositions, postSolverVertexPositions);
    ASSERT_NE(postExternalForcesVertexPositions, postSolverVertexPositions);

    // std::cout << preSolverVertexPositions << std::endl;
    // std::cout << postSolverVertexPositions << std::endl;
}

/*
TEST(Geometry, CuGlBufferSetter)
{
    CuGlBufferSetter<float, GL_ARRAY_BUFFER> vertexBufferSetter;

    std::vector<float> vertexData =
        {
            0, 0, 0,
            10.f, 0, 0,
            0, 10.f, 0,
            0, 0, 10.f};

    std::vector<float> expectedVertexData = {
        0, 0, 0,
        10.f, 0, 0,
        0, 10.f, 0,
        0, 0, 10.f};

    float *d_vertexData = nullptr;
    vertexBufferSetter.allocate(&d_vertexData, vertexData.size() * sizeof(float));
    vertexBufferSetter.copy(d_vertexData, &vertexData[0], vertexData.size());

    cutilSafeCall(
        cudaMemcpy(devPtr, data, nElems * sizeof(T), cudaMemcpyDeviceToHost));

    for (auto [name, geometry] : nameToGeometry)
    {
        std::vector<float> vertexData;
        retrieveProcessedGeom(*geometry, vertexData);

        ASSERT_NE(nameToMeshData.find(name), nameToMeshData.end());

        ASSERT_EQ(vertexData, nameToMeshData[name].m_vertexData);
    }
}
*/

TEST(testTest, test)
{
    EXPECT_EQ(true, true);
}
