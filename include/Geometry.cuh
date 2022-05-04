#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

#include <vector>

#include "Utils/General.h"
#include "Utils/Eigen.h"
#include "CuGlBuffer.cuh"

void __global__ setBufferVals(float setNum,
	float * d_bufferPtr, int bufferSize);

void __global__ addToBufferVertex(float3 setNum,
	float * d_bufferPtr, int bufferSize);

struct Geometry
{
	Geometry(float * h_pBuffer,   			// Host side buffer data
			 size_t bufferElemNum,  		// Number of buffer elements
			 GLuint * _monoColourShader) : 	// Shader id
		monoColourShader(_monoColourShader)
		{ 
		buffer.d_bufferSize = bufferElemNum;
		allocate_cugl_buffer(&buffer);
		set_cugl_buffer(&buffer, h_pBuffer, bufferElemNum);
	}
	Geometry(std::vector<float> * h_Buffer,	// Host side buffer data
			 GLuint * _monoColourShader) : 	// Shader id
		monoColourShader(_monoColourShader)
	{
		buffer.d_bufferSize = h_Buffer->size();
		allocate_cugl_buffer(&buffer);
		set_cugl_buffer(&buffer, h_Buffer);
	}
	cugl_buffer<float> buffer;

    GLuint * monoColourShader;

    ei::Vector3f baseColour = {1,0,0};
};

void drawGeom(Geometry const & geom, Eigen::Matrix4f & cameraMat);

void translateGeom(Geometry & geom, float3 setNum);

#endif /* GEOMETRY_CUH */

/*

GLuint allocateCudaGLBuffer(float * d_bufferPtr, int bufferSize) {

	struct cudaGraphicsResource* verticesVBO_CUDA;

	GLuint verticesVBO;
	glGenBuffers(1, &verticesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, verticesVBO);
	glBufferData(GL_ARRAY_BUFFER, bufferSize * sizeof(float), NULL, GL_DYNAMIC_COPY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&verticesVBO_CUDA,
    							  verticesVBO,
    							  cudaGraphicsMapFlagsWriteDiscard);


	cudaGLRegisterBufferObject(vertexArray);
	
	// Map the buffer to CUDA
	cudaGLMapBufferObject(&d_bufferPtr, vertexArray);
	
	// Run a kernel to create/manipulate the data
	setBufferVals<<<1, bufferSize>>>(2, &d_bufferPtr, bufferSize);
	//MakeVerticiesKernel<<<gridSz,blockSz>>>(&d_bufferPtr, bufferSize);
	
	// Unmap the buffer
	cudaGLUnmapbufferObject(vertexArray);
	return verticesVBO;
}

*/