#include "Utils/Mesh.h"

void generateLine(std::vector<float> & vertices,
                  const ei::Vector3f& origin,
                  const float& length,
                  const Dim& direction){
    const uint stride = 3;
    const uint nVertices = 2;

    vertices.resize(stride * nVertices);
    std::fill(vertices.begin(), vertices.end(), 0);
    vertices[2 + direction] = length;
}

void generateSquare(std::vector<float> & vertices,
                    const ei::Vector3f& origin,
                    const float& squareSize,
                    const std::array<Dim, 2>& dims){
    const uint stride = 3;
    const uint nVertices = 8;
    vertices.resize(stride * nVertices);
    std::fill(vertices.begin(), vertices.end(), 0);

    generateSquarePlane(vertices,
                        origin,
                        squareSize,
                        dims,
                        ei::Vector2ui(1, 1));

    return;
    vertices[0] = 0;
    vertices[1] = 0;
    vertices[2] = 0;
    vertices[3] = 10;
    vertices[4] = 0;
    vertices[5] = 0;

    vertices[6] = 10;
    vertices[7] = 0;
    vertices[8] = 0;
    vertices[9] = 10;
    vertices[10] = 0;
    vertices[11] = 10;

    vertices[12] = 10;
    vertices[13] = 0;
    vertices[14] = 10;
    vertices[15] = 0;
    vertices[16] = 0;
    vertices[17] = 10;

    vertices[18] = 0;
    vertices[19] = 0;
    vertices[20] = 10;
    vertices[21] = 0;
    vertices[22] = 0;
    vertices[23] = 0;
}


void generateSquarePlane(std::vector<float> & vertices,
                         const ei::Vector3f& origin,
                         const float& squareSize,
                         const std::array<Dim, 2>& dims,
                         const ei::Vector2ui& numSquaresPerDim)
{
    const uint stride = 3;
    const uint nVertices = (numSquaresPerDim[0] + 1) * 2 + (numSquaresPerDim[1] + 1) * 2;
    const uint verticesSize = stride * nVertices;
    vertices.resize(verticesSize);

    // Set all vertices to origin

    uint dimStride = 0;
    for (auto[dimIdx1, dimIdx2] : {std::array<uint, 2>{0, 1}, std::array<uint, 2>{1, 0}})
    {
        auto dim1 = dims[dimIdx1];
        const uint& numSquaresDim1 = numSquaresPerDim[dimIdx1] + 1;
        auto dim2 = dims[dimIdx2];
        const uint& numSquaresDim2 = numSquaresPerDim[dimIdx2];
        // Start with 
        for (uint squareIdx=0 ; squareIdx<numSquaresDim1 ; squareIdx++)
        {
            uint strideIdx = dimStride + squareIdx * 6;
            // Vertex 1 set
            uint vertex1StartIdx = strideIdx;
            vertices.at(strideIdx + dim1) += squareIdx * squareSize;
            // Vertex 2 set
            uint vertex2StartIdx = strideIdx + 3;
            vertices.at(vertex2StartIdx + dim1) += squareIdx * squareSize;
            vertices.at(vertex2StartIdx + dim2) += numSquaresDim2 * squareSize;
        }
        dimStride = numSquaresDim1 * 6;
    }
    
}

void generateCubeGrid(std::vector<float>& vertices,
                       const ei::Vector3f& origin,
                       const float& cubeSize,
                       const ei::Vector3ui& numCubesPerDim)
{

    int numCubesX = numCubesPerDim[0] + 1;
    int numCubesY = numCubesPerDim[1] + 1;
    int numCubesZ = numCubesPerDim[2] + 1;

    vertices.resize(
        numCubesX * numCubesY * 6 + 
        numCubesX * numCubesZ * 6 +
        numCubesY * numCubesZ * 6);

    float lineStartX = origin[0];
    float lineStartY = origin[1];
    float lineStartZ = origin[2];

    float lineEndX = origin[0] + numCubesPerDim[0] * cubeSize;
    float lineEndY = origin[1] + numCubesPerDim[1] * cubeSize;
    float lineEndZ = origin[2] + numCubesPerDim[2] * cubeSize;

    int dimStride = 0;
    for (uint i = 0 ; i < numCubesX ; i ++)
        for (uint j = 0 ; j < numCubesY ; j++){
                uint strideIdx = (i * numCubesY + j )* 6;
                vertices.at(strideIdx)     = i * cubeSize + lineStartX;
                vertices.at(strideIdx + 1) = j * cubeSize + lineStartY;
                vertices.at(strideIdx + 2) = lineStartZ;

                vertices.at(strideIdx + 3) = vertices[strideIdx];
                vertices.at(strideIdx + 4) = vertices[strideIdx + 1];
                vertices.at(strideIdx + 5) = lineEndZ;
            }

    dimStride += numCubesX * numCubesY * 6;
    for (uint i = 0 ; i < numCubesX ; i ++)
        for (uint k = 0 ; k < numCubesZ ; k++){
                uint strideIdx = dimStride + (i * numCubesZ + k) * 6;
                vertices.at(strideIdx)     = i * cubeSize + lineStartX;
                vertices.at(strideIdx + 1) = lineStartY;
                vertices.at(strideIdx + 2) = k * cubeSize + lineStartZ;

                vertices.at(strideIdx + 3) = vertices[strideIdx];
                vertices.at(strideIdx + 4) = lineEndY;
                vertices.at(strideIdx + 5) = vertices[strideIdx + 2];
            }

    dimStride += numCubesX * numCubesZ * 6;
    for (uint j = 0 ; j < numCubesY ; j++)
        for (uint k = 0 ; k < numCubesZ ; k++){
            uint strideIdx = dimStride + (j * numCubesZ + k) * 6;
            vertices.at(strideIdx)     = lineStartX;
            vertices.at(strideIdx + 1) = j * cubeSize + lineStartY;
            vertices.at(strideIdx + 2) = k * cubeSize + lineStartZ;

            vertices.at(strideIdx + 3) = lineEndX;
            vertices.at(strideIdx + 4) = vertices[strideIdx + 1];
            vertices.at(strideIdx + 5) = vertices[strideIdx + 2];
        }

}