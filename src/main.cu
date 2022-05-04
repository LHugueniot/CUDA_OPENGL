#include "Camera.h"
#include "GLFWState.h"
#include "MacGrid.cuh"
#include "PlaneGLData.h"
#include "MonoColourGLShader.h"
#include "Geometry.cuh"
#include "CuGlBuffer.cuh"


//struct CUDA_GL_state{
//    int deviceCount;
//    cudaDeviceProp deviceProp;
//    std::vector<int> dev;
//};

//CUDA_GL_state initCUDAGLContext(){
//    CUDA_GL_state state;
//    cudaGetDeviceCount(&state.deviceCount);
//    for (int i=0; i<state.deviceCount; i++) {
//        cudaGetDeviceProperties(&state.deviceProp, state.dev[i]);
//    }
//    // Set Gl device to 1st device (in case of multiple GPUs ?)
//    cudaGLSetGLDevice(state.dev[0]);
//    return state;
//}

int main(int argv, char** args)
{

    std::cout<<"Start of main"<<std::endl;

    //=====================================GLEW/GLFW SETUP=======================================

    std::cout<<"GLEW/GLFW SETUP"<<std::endl;

    // Initialize GLFW
    auto glfwState = setupGLFW(4, 6);

    // Create a GLFW window
    auto mainWindow = createWindow("Main Window");
    auto windowAddedResult = addWindow(glfwState, mainWindow);
	glfwMakeContextCurrent(mainWindow->m_glfwWindow);

    fprintf(stdout, "Adding window result: %i\n", windowAddedResult);
    fprintf(stdout, "m_windows size: %i", glfwState.m_windows.size());

    if (glewInit() != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW.\n");
        return EXIT_FAILURE;
    }

    //=====================================CUDA SETUP===========================================
    // Initialize CUDA context (on top of the GL context)

    std::cout<<"CUDA SETUP"<<std::endl;

    cudaSetDevice(0);
    cudaGLSetGLDevice(0);

    //=====================================OPENGL SETUP=========================================
    
    std::cout<<"OPENGL SETUP"<<std::endl;

    glClearColor(0.5f, 0.5f, 0.5f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //glEnable(GL_DEPTH_TEST);
    //glDepthFunc(GL_LESS);

    //glfwSwapBuffers(mainWindow->m_glfwWindow);

    //=====================================CAMERA SETUP=========================================

    std::cout<<"CAMERA SETUP"<<std::endl;

    auto camera = Camera(mainWindow->m_windowWidth, mainWindow->m_windowHeight);

    rotateCamera(camera, TO_RAD(-45.f));
    pitchCamera(camera,  TO_RAD(-45.f));

    updateCamera(camera);

    //=====================================SHADER SETUP=========================================

    std::cout<<"SHADER SETUP"<<std::endl;
    
    GLuint monoColourShader = compileMonoColourShaderProgram();
    if (monoColourShader == 0)
    {
        fprintf(stderr, "Shader setup failed.\n");
        return EXIT_FAILURE;
    }

    //=====================================MESH DATA SETUP====================================

    std::cout<<"MESH DATA SETUP"<<std::endl;
    
    //Create center of world grid plain
    std::vector<float> gridPlaneVertexData;
    generateTile(gridPlaneVertexData);
    generateLine(gridPlaneVertexData);
    generatePlaneVertexData(gridPlaneVertexData, 1, 6, 6);
    PlaneGLData gridPlane(&gridPlaneVertexData, &monoColourShader);
    initPlaneVAO(gridPlane);

    Geometry test_geom(&gridPlaneVertexData, &monoColourShader);

    std::cout<<"starting loop"<<std::endl;

    glfwMakeContextCurrent(mainWindow->m_glfwWindow);
    glViewport(0, 0, mainWindow->m_windowWidth, mainWindow->m_windowHeight);

    while(!shouldQuit(glfwState))
    {
        fprintf(stdout, "Drawing for window: %s\n", mainWindow->m_windowTitle.c_str());


        for (auto window : glfwState.m_windows)
        {
            fprintf(stdout, "Drawing for window: %s\n", window->m_windowTitle.c_str());
            glfwMakeContextCurrent(window->m_glfwWindow);

            glClearColor(1.f, .5f, .5f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            updateCamera(camera);
            ei::Matrix4f cameraVP = camera.projMat * camera.viewMat;
            // Main stuff
            updatePlaneVAO(gridPlane);
            drawPlane(gridPlane, cameraVP);
            drawGeom(test_geom, cameraVP);


            glfwSwapBuffers(window->m_glfwWindow);
            glfwPollEvents();
        }
    }


    //while(!quit)
    //{
    //    std::cout<<"tick"<<std::endl;
    //    SDL_Event event;
    //    if (SDL_PollEvent(&event) != 0){
    //        switch (event.type){
    //            case SDL_QUIT:
    //                quit = true;
    //                break;
//
    //            case SDL_KEYDOWN:
    //                switch(event.key.keysym.sym){
    //                case SDLK_LEFT:
    //                    moveCamera(camera, Camera::ORBIT_LEFT);
    //                    std::cout<<"SDLK_LEFT"<<std::endl;
    //                    break;
    //                case SDLK_RIGHT:
    //                    moveCamera(camera, Camera::ORBIT_RIGHT);
    //                    std::cout<<"SDLK_RIGHT"<<std::endl;
    //                    break;
    //                case SDLK_UP:
    //                    moveCamera(camera, Camera::ORBIT_UP);
    //                    std::cout<<"SDLK_UP"<<std::endl;
    //                    break;
    //                case SDLK_DOWN:
    //                    moveCamera(camera, Camera::ORBIT_DOWN);
    //                    std::cout<<"SDLK_DOWN"<<std::endl;
    //                    break;
    //                case SDLK_e:{
    //                    spag::Geometry test_geom2(&gridPlaneVertexData, &monoColourShader);
    //                    
    //                }
    //                    break;
//
    //                case SDLK_w:
    //                    //spag::translateGeom(test_geom, float3{0,0.0001,0});
    //                    spag::addToBufferVertex<<<1, static_cast<int>((float)test_geom.buffer.d_bufferSize/3.f)>>>
    //                        (float3{0,0.01,0}, test_geom.buffer.d_pBuffer, test_geom.buffer.d_bufferSize);
    //                    break;
    //                }
    //                break;
//
    //            case SDL_MOUSEWHEEL:
    //                if(event.wheel.y < 0)       // scroll up
    //                    moveCamera(camera, Camera::ZOOM_IN);
    //                else if(event.wheel.y > 0)  // scroll down
    //                    moveCamera(camera, Camera::ZOOM_OUT);
    //                break;
    //        }
    //    }
//
    //    glClearColor(.5f, 0.5f, 0.5f, 1.f);
    //    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
    //    updateCamera(camera);
    //    Matrix4f cameraVP = camera.projMat * camera.viewMat;
//
    //    // Main stuff
    //    updatePlaneVAO(gridPlane);
    //    drawPlane(gridPlane, cameraVP);
//
    //    drawGeom(test_geom, cameraVP);
//
    //    glColor3f(0.0f,0.0f,1.0f); //blue color
    //    glBegin(GL_POINTS); //starts drawing of points
    //        glVertex3f(1.0f,1.0f,0.0f);//upper-right corner
    //        glVertex3f(-1.0f,-1.0f,0.0f);//lower-left corner
    //    glEnd();//end drawing of points
//
    //    SDL_GL_SwapWindow(state_sdl.window);
    //}
    //
    teardown(glfwState);
    return EXIT_SUCCESS;
}