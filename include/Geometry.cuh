#include "Utils.cuh"
#include "CuGlBuffer.cuh"

namespace spag{

void __global__ setBufferVals(float setNum,
	float * d_bufferPtr, int bufferSize){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < bufferSize)
		d_bufferPtr[idx] = idx;
}

void __global__ addToBufferVertex(float3 setNum,
	float * d_bufferPtr, int bufferSize){

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
    int idx_x = idx;
    int idx_y = idx + 1;
    int idx_z = idx + 2;

	if(idx_x < bufferSize)
		d_bufferPtr[idx_x] += setNum.x;
	if(idx_y < bufferSize)
		d_bufferPtr[idx_y] += setNum.y;
	if(idx_z < bufferSize)
		d_bufferPtr[idx_z] += setNum.z;
}

struct Geometry{
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
		monoColourShader(_monoColourShader){
		buffer.d_bufferSize = h_Buffer->size();
		allocate_cugl_buffer(&buffer);
		set_cugl_buffer(&buffer, h_Buffer);
	}
	cugl_buffer<float> buffer;

    GLuint * monoColourShader;

    Vector3f baseColour = {1,0,0};
};

void drawGeom(Geometry const & geom, Eigen::Matrix4f & cameraMat){

	//std::cout<<"DEBUG 1"<<std::endl;
	glUseProgram(*geom.monoColourShader);
    //std::cout<<"DEBUG 1.25"<<std::endl;
    GLuint mvpID = glGetUniformLocation(*geom.monoColourShader, "MVP");
    //std::cout<<"DEBUG 1.5"<<std::endl;
    glUniformMatrix4fv(mvpID, 1, GL_FALSE, cameraMat.data());

//std::cout<<"DEBUG 2"<<std::endl;
    GLuint baseColID = glGetUniformLocation(*geom.monoColourShader, "base_colour");
    glUniform3fv(baseColID, 1, geom.baseColour.data());

//std::cout<<"DEBUG 3"<<std::endl;
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, geom.buffer.gl_VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(GL_LINES, 0, geom.buffer.d_bufferSize/3);
    glDisableVertexAttribArray(0);
}

void translateGeom(Geometry & geom, float3 setNum){

	auto & d_pBuffer = geom.buffer.d_pBuffer;
	auto & cugl_pVBO = geom.buffer.cugl_pVBO;
	size_t bufferSize = geom.buffer.d_bufferSize;
	size_t bufferSizeBytes;

    // Map buffer object
    cutilSafeCall(
	cudaGraphicsMapResources(1, &cugl_pVBO)
	);

	// Get pointer to use, not sure if possible to use outside of mapped scope
	cutilSafeCall(
	cudaGraphicsResourceGetMappedPointer((void**)&d_pBuffer,
                                         &bufferSizeBytes,
                                         cugl_pVBO)
	);

    spag::addToBufferVertex<<<1, static_cast<int>((float)bufferSize/3.f)>>>
        (setNum, d_pBuffer, bufferSize);

    // Unmap buffer object
    cutilSafeCall(cudaGraphicsUnmapResources(1, &cugl_pVBO));
}

}


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