#include "gtest/gtest.h"
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include "Camera.h"
#include "Geometry.cuh"
#include "LoadGeometry.cuh"

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>


template<typename T, typename P=float>
bool _EXPECT_NEAR(T a, T b, P p)
{
    EXPECT_NEAR(a, b, p);
    T diff = std::abs(a - b);
    return p > diff && diff >= 0;
}

template<typename T, typename P=float>
void EXPECT_NEAR_STDVEC(T a, T b, P p=0.00001)
{
    ASSERT_EQ(a.size(), b.size());
    bool val = true;
    for (int i=0 ; i<a.size() ; i++)
    {
        val &= _EXPECT_NEAR(a.data()[i], b.data()[i], p);
    }
    if (!val)
    {
        EXPECT_EQ(a, b);
    }
}

template<typename T, typename P=float>
void EXPECT_NEAR_MATRIX(T a, T b, P p=0.00001)
{
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    bool val = true;
    for (int i=0 ; i<a.rows() * a.cols() ; i++)
    {
        val &= _EXPECT_NEAR(a.data()[i], b.data()[i], p);
    }
    if (!val)
    {
        EXPECT_EQ(a, b);
    }
}

void print_glm_mat4(glm::mat4 & mat)
{
    for (int r=0; r<4; r++){
        for (int c=0; c<4; c++)
        {
            fprintf(stdout, "%7.1f", mat[r][c]);
        }
        fprintf(stdout, "\n");
    }
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
        glm::radians(radians_fov), windowWidth / windowHeight, near, far
    );
    glm::mat4 glm_view = glm::lookAt(
        glm::vec3(0, 0, 30), // Camera is at (4,3,-3), in World Space
        glm::vec3(0, 0, 0), // and looks at the origin
        glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
	);

    glm::mat4 glm_PV = glm_projection * glm_view;

    ei::Matrix4f ei_projection;
    ei::Matrix4f ei_view;

    ei_utils::setProjMat(ei_projection, windowWidth, windowHeight, TO_RAD(fov), far, near);
    ei_utils::setLookAt(ei_view, {0, 0, 30}, {0, 0, 0}, {0, 1, 0});

    //ei::Matrix4f ei_PV = ei_projection.transpose() * ei_view.transpose();
    ei::Matrix4f ei_PV = ei_projection * ei_view;

    ASSERT_EQ(TO_RAD(fov), glm::radians(radians_fov));

    for (int i=0; i<16; i++)
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

void setupTestGeom(Geometry& testGeom,
                   std::vector<float>& vertexData)
{
    testGeom.d_vertexPositionBufferSize = vertexData.size();

    size_t bufferBytesSize = testGeom.d_vertexPositionBufferSize *
                             sizeof(float);

    cudaMalloc((void**) &testGeom.d_vertexPositionBufferData, 
               bufferBytesSize);

    cudaMemcpy(testGeom.d_vertexPositionBufferData,
               &vertexData.data()[0],
               bufferBytesSize,
               cudaMemcpyHostToDevice);
}

void retriveProcessedGeom(Geometry& testGeom,
                          std::vector<float>& vertexData)
{
    size_t bufferBytesSize = testGeom.d_vertexPositionBufferSize *
                             sizeof(float);

    cudaMemcpy(&vertexData.data()[0],
               testGeom.d_vertexPositionBufferData,
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
        0, 0, 1
    };
    
    ei::Vector3f scale = {2.5f, 5.0f, 10.0f};
    setupTestGeom(testGeom, vertexData);

    scaleGeom(testGeom, scale);

    retriveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
    {
        0, 0, 0,
        2.5f, 0, 0,
        0, 5.0f, 0,
        0, 0, 10.0f
    };

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
        0, 0, 1
    };
    
    ei::Vector3f scale = {5.f, 5.f, 5.f};
    setupTestGeom(testGeom, vertexData);

    scaleGeom(testGeom, scale);

    retriveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
    {
        0, 0, 0,
        5.f, 0, 0,
        0, 5.f, 0,
        0, 0, 5.f
    };

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
        0, 0, 10.f
    };

    ei::Vector3f scale = {2.f, 2.f, 2.f};
    ei::Vector3f pivot = {5.f, 5.f, 5.f};
    setupTestGeom(testGeom, vertexData);

    scaleGeom(testGeom, scale, pivot);

    retriveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
    {
        -5, -5, -5,
        15, -5, -5,
        -5, 15, -5,
        -5, -5, 15
    };

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
        0, 0, 10.f
    };

    ei::Vector3f axis = {0, 1.f, 0};

    setupTestGeom(testGeom, vertexData);

    rotateGeom(testGeom, axis, TO_RAD(90), {0, 0, 0});

    retriveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
    {
        0, 0, 0,
        0, 0, 10.f,
        0, 10.f, 0,
        -10.f, 0, 0
    };

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
        0, 0, 10.f
    };

    ei::Vector3f axis = {0, 1.f, 0};

    setupTestGeom(testGeom, vertexData);

    rotateGeom(testGeom, axis, TO_RAD(90), {5.f, 0, 0});

    retriveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
    {
        5.f, 0, -5.f,
        5.f, 0, 5.f,
        5.f, 10.f, -5.f,
        -5.f, 0, -5.f
    };

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
        0, 0, 10.f
    };

    ei::Vector3f translation = {-10.f, -10.f, -10.f};
    setupTestGeom(testGeom, vertexData);

    translateGeom(testGeom, translation);

    retriveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
    {
        -10.f, -10.f, -10.f,
        0.f, -10.f, -10.f,
        -10.f, 0.f, -10.f,
        -10.f, -10.f, 0.f
    };

    EXPECT_NEAR_STDVEC(vertexData, expectedVertexData);
}

TEST(Geometry, transformGeom1)
{
    auto testGeom = Geometry();
    std::vector<float> vertexData =
    {
        1, 1, 1
    };

    ei::Transform3f transform = ei::Translation3f({10.f, 10.f, 10.f}) *
                                ei::AngleAxisf(TO_RAD(90), ei::Vector3f{0.f, 1.f, 0.f}) *
                                ei::Scaling(10.f, 10.f, 10.f);

    setupTestGeom(testGeom, vertexData);

    transformGeom(testGeom, transform);

    retriveProcessedGeom(testGeom, vertexData);

    std::vector<float> expectedVertexData =
    {
        -110.f, 110.f, -110.f
    };

    EXPECT_NEAR_STDVEC(vertexData, expectedVertexData, 0.001);
}

TEST(Geometry, LoadGeometry)
{
    const aiScene* sceneCache = nullptr;

    std::filesystem::path assetFile(__FILE__);
    
    assetFile = std::filesystem::absolute(
        assetFile.parent_path() / ".." / "assets" / "PantherBoss" / "PAN.obj");
    std::cout<<assetFile<<std::endl;

    std::vector<const aiMesh*> meshes = loadAiMeshes(assetFile, &sceneCache);

    std::cout<<"Num meshes: "<<meshes.size()<<std::endl;

    auto allocator = DefaultCudaAllocator();
    auto copier = DefaultCudaCopyToDevice();
    std::vector<std::pair<std::string, Geometry*>> nameToGeometry =
        initGeometryFromAiMeshes<Geometry>(
            meshes,
            allocator,
            copier);

    std::cout<<"Size of name to geom: "<<nameToGeometry.size()<<std::endl;

    for (auto [name, geometry] : nameToGeometry)
    {
        std::cout<<name<<std::endl;
    }

    aiReleaseImport(sceneCache);
}

TEST(testTest, test)
{
    EXPECT_EQ(true, true);
}
