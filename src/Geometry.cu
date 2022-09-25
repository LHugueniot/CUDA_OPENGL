#include "Geometry.cuh"

// ================================================================================================

void __global__
scaleVertices(const ei::Vector3f scale,
              const ei::Vector3f pivot,
	          float * vertexBufferData,
              uint vertexBufferSize)
{
    uint idx = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
    uint endIdx = idx + 2;

    if (!(endIdx < vertexBufferSize))
        return;

    Eigen::Map<ei::Vector3f> vertex(&vertexBufferData[idx]);
    ei::Vector3f diff = (vertex - pivot);
    diff = {
        diff.x() * scale.x(),
        diff.y() * scale.y(),
        diff.z() * scale.z()
    };
    vertex = diff + pivot;
}

void scaleGeom(Geometry& geom,
               const ei::Vector3f& scale,
               const ei::Vector3f& pivot)
{
    assert(geom.d_vertexPositionBufferSize % 3 == 0);
    scaleVertices<<<1, static_cast<int>((float)geom.d_vertexPositionBufferSize/3.f)>>>
        (scale, pivot, geom.d_vertexPositionBufferData, geom.d_vertexPositionBufferSize);
}

// ================================================================================================

void __global__
rotateVertices(const ei::Matrix3f rotation,
               const ei::Vector3f pivot,
	           float * vertexBufferData,
               uint vertexBufferSize)
{
    uint idx = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
    uint endIdx = idx + 2;

    if (!(endIdx < vertexBufferSize))
        return;

    ei::Map<ei::Vector3f> vertex(&vertexBufferData[idx]);

    vertex -= pivot;
    vertex = vertex.transpose() * rotation;
    vertex += pivot;
}

void rotateGeom(Geometry& geom,
                const ei::Vector3f& axis,
                const float angle,
                const ei::Vector3f& pivot)
{
    assert(geom.d_vertexPositionBufferSize % 3 == 0);
    ei::Transform3f transform(ei::AngleAxis<float>(angle, axis.normalized()));

    rotateVertices<<<1, static_cast<int>((float)geom.d_vertexPositionBufferSize/3.f)>>>
        (transform.rotation(), pivot, geom.d_vertexPositionBufferData, geom.d_vertexPositionBufferSize);
}

// ================================================================================================

void __global__
translateVertices(const ei::Vector3f translation,
	              float * vertexBufferData,
                  uint vertexBufferSize)
{
    uint idx = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
    uint endIdx = idx + 2;

    if (!(endIdx < vertexBufferSize))
        return;

    Eigen::Map<ei::Vector3f> vertex(&vertexBufferData[idx]);
    vertex += translation;
}

void translateGeom(Geometry& geom,
                   const ei::Vector3f& translation)
{
    assert(geom.d_vertexPositionBufferSize % 3 == 0);
    translateVertices<<<1, static_cast<int>((float)geom.d_vertexPositionBufferSize/3.f)>>>
        (translation, geom.d_vertexPositionBufferData, geom.d_vertexPositionBufferSize);
}

// ================================================================================================

void __global__
transformVertices(const ei::Vector3f translation,
                  const ei::Matrix3f rotation,
                  const ei::Matrix3f scaling,
	              float * vertexBufferData,
                  uint vertexBufferSize)
{
    uint idx = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
    uint endIdx = idx + 2;

    if (!(endIdx < vertexBufferSize))
        return;

    ei::Map<ei::Vector3f> vertex(&vertexBufferData[idx]);

    vertex = (vertex + translation).transpose() * rotation * scaling;
}

void transformGeom(Geometry& geom,
                   const ei::Transform3f& transform)
{
    assert(geom.d_vertexPositionBufferSize % 3 == 0);
    transformVertices<<<1, static_cast<int>((float)geom.d_vertexPositionBufferSize/3.f)>>>(
        transform.translation(),
        transform.rotation(),
        transform.linear(),
        geom.d_vertexPositionBufferData,
        geom.d_vertexPositionBufferSize);
}

// ================================================================================================
