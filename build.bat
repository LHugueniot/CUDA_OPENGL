nvcc -std=c++17 ^
 	src\Camera.cpp  ^
	src\CuGlBuffer.cu  ^
	src\Geometry.cu  ^
	src\GLShader.cpp  ^
	src\main.cu  ^
	src\MonoColourGLShader.cpp  ^
	src\PlaneGLData.cpp ^
	-I%WS_PATH%\glew-2.1.0-win32\glew-2.1.0\include ^
	-I%WS_PATH%\SDL2-devel-2.0.20-VC\SDL2-2.0.20\include ^
 	-I%WS_PATH%\eigen ^
 	-Iinclude ^
 	-L%WS_PATH%\glew-2.1.0-win32\glew-2.1.0\
 	-lglew32 ^
 	-lGL ^
 	-lSDL2 ^
 	-lcudart ^
 	-DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS ^
 	-o build\a.out