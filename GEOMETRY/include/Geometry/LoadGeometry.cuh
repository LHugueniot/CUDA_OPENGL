#ifndef LOAD_GEOMETRY_CUH
#define LOAD_GEOMETRY_CUH

#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "Geometry/Buffer/BufferSetters.cuh"
#include "Geometry/Geometry.cuh"

#include "Utils/Assimp.h"
#include "Utils/Cuda.cuh"
#include "Utils/OpenGL.h"

constexpr uint kDefaultImportFlags = \
    aiProcess_Triangulate |
    aiProcess_JoinIdenticalVertices |
    aiProcess_ImproveCacheLocality |
    aiProcess_FindInvalidData;

// constexpr uint kDefaultImportFlags = 0;


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
        printf("Num vertices %i", meshPtr->mNumVertices);
        geom->d_nVertexPositionBufferElems = meshPtr->mNumVertices * 3;
        vertexBufferSetter.allocate(&geom->d_vertexPositionBufferData,
                                    geom->d_nVertexPositionBufferElems);

        // Copy over vertex data
        float *vertexData = &(meshPtr->mVertices[0].x);
        vertexBufferSetter.copy(geom->d_vertexPositionBufferData, vertexData,
                                geom->d_nVertexPositionBufferElems);

        uint nFaceIdxElems = meshPtr->mNumFaces * 3;
        uint *faceIdxData = new uint[nFaceIdxElems];

        std::set<std::pair<uint, uint>> uniqueEdgeIdxs;

        for (uint t = 0; t < meshPtr->mNumFaces; ++t)
        {
            const aiFace *face = &meshPtr->mFaces[t];

            assert(face->mNumIndices == 3);

            for (uint vertIdx = 0; vertIdx < face->mNumIndices; vertIdx++)
            {
                uniqueEdgeIdxs.insert(
                    makeOrderedIdxPair(face->mIndices[vertIdx],
                                       face->mIndices[(vertIdx + 1) % face->mNumIndices]));

                faceIdxData[t * 3 + vertIdx] = face->mIndices[vertIdx];
            }
        }

        uint nEdgeIdxElems = uniqueEdgeIdxs.size() * 2;
        uint *edgeIdxData = new uint[nEdgeIdxElems];

        uint i = 0;
        for (const auto &edge : uniqueEdgeIdxs)
        {
            edgeIdxData[i * 2] = edge.first;
            edgeIdxData[i * 2 + 1] = edge.second;
            i++;
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
