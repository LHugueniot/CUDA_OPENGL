#ifndef LOAD_GEOMETRY_CUH
#define LOAD_GEOMETRY_CUH

#include "Geometry.cuh"

#include "Utils/Assimp.h"

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>

std::vector<const aiMesh*> loadAiMeshes(
    const std::filesystem::path& sceneFilePath, const aiScene** sceneCachePtr);

//template<typename T>
//using cudaAllocatorSig = void (*)(T**, size_t);

struct DefaultCudaAllocator
{
    template<typename T>
    void operator()(T** devPtr, size_t size)
    {
        cudaMalloc((void**) &devPtr, size);
    }
};

struct DefaultCudaCopyToDevice
{
    template<typename T>
    void operator()(T* devPtr, T* data, size_t size)
    {
        cudaMemcpy(devPtr, data, size, cudaMemcpyHostToDevice);
    }
};

template <typename Geometry_t,
          typename allocatorFunc_t,
          typename copierFunc_t>
void initGeometryFromAiMeshesDummy(
    allocatorFunc_t&& allocatorFunc=DefaultCudaAllocator(),
    copierFunc_t&& copierFunc=DefaultCudaCopyToDevice())
{}

template <typename Geometry_t,
          typename allocatorFunc_t,
          typename copierFunc_t>
std::vector<std::pair<std::string, Geometry_t*>> initGeometryFromAiMeshes(
    const std::vector<const aiMesh*>& meshes,
    allocatorFunc_t&& allocatorFunc=DefaultCudaAllocator(),
    copierFunc_t&& copierFunc=DefaultCudaCopyToDevice())
{
    std::vector<std::pair<std::string, Geometry_t*>> geometries;
    geometries.reserve(meshes.size());
    for (const aiMesh* meshPtr : meshes)
    {
        assert(meshPtr);
        // Assert the mesh is triangulated, we do not currently support
        // polygons
        assert(meshPtr->mPrimitiveTypes & aiPrimitiveType_TRIANGLE);

        Geometry_t * geom = new Geometry_t();
        geometries.emplace_back(meshPtr->mName.data, geom);

        // Allocate device memory for vertices
        geom->d_vertexPositionBufferSize = meshPtr->mNumVertices * 3 * sizeof(float);
        allocatorFunc(&geom->d_vertexPositionBufferData, geom->d_vertexPositionBufferSize);

        // Copy over vertex data
        float * vertexData = &(meshPtr->mVertices[0].x);
        copierFunc(geom->d_vertexPositionBufferData,
                                vertexData,
                                geom->d_vertexPositionBufferSize);

        uint* edgeIdxData = new uint[meshPtr->mNumFaces * 6];
        uint* faceIdxData = new uint[meshPtr->mNumFaces * 3];
		for (uint t = 0; t < meshPtr->mNumFaces; ++t)
        {
			const aiFace* face = &meshPtr->mFaces[t];

            for (uint vertIdx=0; vertIdx<face->mNumIndices; vertIdx++)
            {
                edgeIdxData[t * 6 + vertIdx * 2] = face->mIndices[vertIdx];
                edgeIdxData[t * 6 + vertIdx * 2 + 1] = face->mIndices[(vertIdx + 1) % face->mNumIndices];
                faceIdxData[t * 3 + vertIdx] = face->mIndices[vertIdx];
            }
        }

        // Allocate device memory for edges
        geom->d_edgeIdxBufferSize = meshPtr->mNumFaces * 6 * sizeof(uint);
        allocatorFunc(&geom->d_edgeIdxBufferData, geom->d_edgeIdxBufferSize);
        // Copy over edge data
        copierFunc(geom->d_edgeIdxBufferData,
                                edgeIdxData,
                                geom->d_edgeIdxBufferSize);

        // Allocate device memory for edges
        geom->d_triangleIdxBufferSize = meshPtr->mNumFaces * 3 * sizeof(uint);
        allocatorFunc(&geom->d_triangleIdxBufferData, geom->d_triangleIdxBufferSize);
        // Copy over edge data
        copierFunc(geom->d_triangleIdxBufferData,
                                faceIdxData,
                                geom->d_triangleIdxBufferSize);
    }
    return geometries;
}

#endif /* LOAD_GEOMETRY_CUH */
