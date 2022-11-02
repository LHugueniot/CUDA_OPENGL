#include "Geometry/LoadGeometry.cuh"

std::vector<const aiMesh *>
loadAiMeshes(const std::filesystem::path &sceneFilePath,
             const aiScene **sceneCachePtr, uint importFlags)
{
    std::ostringstream errStream;

    bool fileExists = std::filesystem::exists(sceneFilePath);
    if (!fileExists)
    {
        errStream << "File:" << sceneFilePath << " could not be found."
                  << std::endl;
        throw std::exception(errStream.str().c_str());
    }
    auto sceneCache = aiImportFile(sceneFilePath.u8string().c_str(), importFlags);

    // Assimp::Importer importer;
    // auto sceneCache = importer.ReadFile(sceneFilePath.string().c_str(),
    // aiProcess_Triangulate);
    *sceneCachePtr = sceneCache;

    if (!sceneCache)
    {
        errStream << "Could not open file: " << sceneFilePath
                  << " Reason: " << aiGetErrorString() << std::endl;
        // errStream<<"Could not open file: "<<sceneFilePath<<" Reason:
        // "<<importer.GetErrorString()<<std::endl;
        throw std::exception(errStream.str().c_str());
    }

    auto rootNode = sceneCache->mRootNode;
    std::vector<const aiMesh *> result;

    if (uint numMeshes = sceneCache->mNumMeshes)
    {
        result.reserve(numMeshes);

        for (uint n = 0; n < numMeshes; ++n)
        {
            const aiMesh *meshPtr = sceneCache->mMeshes[n];
            assert(meshPtr != nullptr);
            result.push_back(meshPtr);
        }
    }
    // else if (uint numMeshes = sceneCache->mRootNode->mNumMeshes)
    //{
    //     result.reserve(numMeshes);
    //
    //    for (uint n=0 ; n < numMeshes; ++n) {
    //        auto mesh =
    //        sceneCache->mRootNode->mMeshes[sceneCache->mRootNode->mMeshes[n]];
    //        assert(mesh != nullptr);
    //        result.push_back(mesh);
    //    }
    //}

    return result;
}
