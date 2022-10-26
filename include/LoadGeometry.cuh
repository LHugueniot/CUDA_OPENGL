#ifndef LOAD_GEOMETRY_CUH
#define LOAD_GEOMETRY_CUH

#include "Geometry.cuh"

#include "Utils/Assimp.h"
#include "Utils/Cuda.cuh"
#include "Utils/OpenGL.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

constexpr uint kDefaultImportFlags =
    aiProcess_Triangulate | aiProcess_JoinIdenticalVertices;

template <typename T, GLenum GLBufferType = GL_ARRAY_BUFFER>
struct CuGlBufferSetter
{
    void allocate(T **devPtr, size_t nElems)
    {
        m_nElements = nElems;
        m_sizeOfElement = sizeof(T);

        size_t bufferSize = m_nElements * m_sizeOfElement;

        glGenBuffers(1, &m_glBufferId);
        checkGLError();
        glBindBuffer(GLBufferType, m_glBufferId);
        checkGLError();
        glBufferData(GLBufferType, m_nElements * m_sizeOfElement, nullptr,
                     GL_DYNAMIC_DRAW);
        checkGLError();

        // Map buffer object
        cutilSafeCall(cudaGraphicsGLRegisterBuffer(&m_resourceObj, m_glBufferId,
                                                   cudaGraphicsMapFlagsWriteDiscard));

        // Map buffer object
        cutilSafeCall(cudaGraphicsMapResources(1, &m_resourceObj, 0));

        size_t size;
        // Get pointer to use, not sure if possible to use outside of mapped scope
        cutilSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void **>(devPtr), &size, m_resourceObj));
        assert(size == (m_nElements * m_sizeOfElement));
    }

    void copy(T *devPtr, T *data, size_t nElems)
    {
        size_t bufferSize = nElems * sizeof(T);
        m_data.resize(nElems);
        memcpy(&m_data.data()[0], data, bufferSize);

        // Copy copy data from host to device buffer
        cutilSafeCall(
            cudaMemcpy(devPtr, data, nElems * sizeof(T), cudaMemcpyHostToDevice));
        // Unmap buffer object
        cudaGraphicsUnmapResources(1, &m_resourceObj, 0);

        glBindBuffer(GLBufferType, 0);
        checkGLError();
    }

    std::vector<T> m_data = {};

    GLuint m_glBufferId = 0;

    size_t m_nElements = 0;
    size_t m_sizeOfElement = 0;

    struct cudaGraphicsResource *m_resourceObj = nullptr;
};

template <typename T>
struct DefaultCudaBufferSetter
{
    void allocate(T **devPtr, size_t nElems)
    {
        cutilSafeCall(
            cudaMalloc(reinterpret_cast<void **>(devPtr), nElems * sizeof(T)));
    }

    void copy(T *devPtr, T *data, size_t nElems)
    {
        size_t bufferSize = nElems * sizeof(T);
        m_data.resize(nElems);
        memcpy(&m_data.data()[0], data, bufferSize);
        cutilSafeCall(cudaMemcpy(devPtr, data, bufferSize, cudaMemcpyHostToDevice));
    }
    std::vector<T> m_data;
};

std::vector<const aiMesh *>
loadAiMeshes(const std::filesystem::path &sceneFilePath,
             const aiScene **sceneCachePtr,
             uint importFlags = kDefaultImportFlags);

template <typename Geometry_t,
          typename Setter_1 = DefaultCudaBufferSetter<float>,
          typename Setter_2 = DefaultCudaBufferSetter<uint>,
          typename Setter_3 = DefaultCudaBufferSetter<uint>>
std::vector<std::pair<std::string, Geometry_t *>>
initGeometryFromAiMeshes(const std::vector<const aiMesh *> &meshes,
                         Setter_1 &&vertexBufferSetter = {},
                         Setter_2 &&edgeBufferSetter = {},
                         Setter_3 &&faceBufferSetter = {})
{
    std::vector<std::pair<std::string, Geometry_t *>> geometries;
    geometries.reserve(meshes.size());
    for (const aiMesh *meshPtr : meshes)
    {
        assert(meshPtr);
        // Assert the mesh is triangulated, we do not currently support
        // polygons
        assert(meshPtr->mPrimitiveTypes & aiPrimitiveType_TRIANGLE);

        Geometry_t *geom = new Geometry_t();
        geometries.emplace_back(meshPtr->mName.data, geom);

        // Allocate device memory for vertices
        geom->d_nVertexPositionBufferElems = meshPtr->mNumVertices * 3;
        vertexBufferSetter.allocate(&geom->d_vertexPositionBufferData,
                                    geom->d_nVertexPositionBufferElems);

        // Copy over vertex data
        float *vertexData = &(meshPtr->mVertices[0].x);
        vertexBufferSetter.copy(geom->d_vertexPositionBufferData, vertexData,
                                geom->d_nVertexPositionBufferElems);

        uint nEdgeIdxElems = meshPtr->mNumFaces * 6;
        uint *edgeIdxData = new uint[nEdgeIdxElems];

        uint nFaceIdxElems = meshPtr->mNumFaces * 3;
        uint *faceIdxData = new uint[nFaceIdxElems];

        for (uint t = 0; t < meshPtr->mNumFaces; ++t)
        {
            const aiFace *face = &meshPtr->mFaces[t];

            for (uint vertIdx = 0; vertIdx < face->mNumIndices; vertIdx++)
            {
                edgeIdxData[t * 6 + vertIdx * 2] = face->mIndices[vertIdx];
                edgeIdxData[t * 6 + vertIdx * 2 + 1] =
                    face->mIndices[(vertIdx + 1) % face->mNumIndices];

                faceIdxData[t * 3 + vertIdx] = face->mIndices[vertIdx];
            }
        }

        // Allocate device memory for edges
        geom->d_nEdgeIdxBufferElems = nEdgeIdxElems;
        edgeBufferSetter.allocate(&geom->d_edgeIdxBufferData,
                                  geom->d_nEdgeIdxBufferElems);
        // Copy over edge data
        edgeBufferSetter.copy(geom->d_edgeIdxBufferData, edgeIdxData,
                              geom->d_nEdgeIdxBufferElems);

        // Allocate device memory for edges
        geom->d_nTriangleIdxBufferElems = nFaceIdxElems;
        faceBufferSetter.allocate(&geom->d_triangleIdxBufferData,
                                  geom->d_nTriangleIdxBufferElems);
        // Copy over edge data
        faceBufferSetter.copy(geom->d_triangleIdxBufferData, faceIdxData,
                              geom->d_nTriangleIdxBufferElems);
    }
    return geometries;
}

#endif /* LOAD_GEOMETRY_CUH */
