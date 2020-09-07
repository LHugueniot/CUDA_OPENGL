#include "MacGrid.cuh"
#include "PlaneGLData.h"
#include "Camera.h"
#include "MonoColourGLShader.h"
#include "Geometry.cuh"
#include "CuGlBuffer.cuh"

static uint windowWidth = 1920;
static uint windowHeight = 1080;

//extern "C" void initPressure(MacGrid grid, int newPressure);


struct SDL_state
{
    SDL_Window* window = nullptr;
    SDL_GLContext context = nullptr;
};

void teardown(SDL_state state)
{
    SDL_GL_DeleteContext(state.context);
    SDL_DestroyWindow(state.window);
    SDL_Quit();
}

SDL_state setupSDL()
{
    SDL_state state;

    if (SDL_Init(SDL_INIT_VIDEO)<0 ){
        std::cout<<"Failed to init SDL: "<<SDL_GetError()<<std::endl;
        return state;
    }
    std::atexit(SDL_Quit);

    //SDL_GL_LoadLibrary(0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, 
               SDL_GL_CONTEXT_PROFILE_CORE);

    //SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    //SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    //SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    //SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    //SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

    state.window = SDL_CreateWindow("Point Viewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                    windowWidth, windowHeight, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if(!state.window){
        std::cout<<"Failed to create window: "<<SDL_GetError()<<std::endl;
        return state;
    }

    state.context = SDL_GL_CreateContext(state.window);
    if(!state.context){
        std::cout<<"Failed to create GL context: "<<SDL_GetError()<<std::endl;
        return state;
    }
    return state;
    //SDL_GL_Create_
}

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

int main(void) {

    //=====================================GLEW/SDL SETUP=======================================
    // Initialize SDL
    auto state_sdl = setupSDL();

    // Initialize GLEW
    if (glewInit() != GLEW_OK) exit(EXIT_FAILURE); // GLEW failed!

    //=====================================CUDA SETUP===========================================
    // Initialize CUDA context (ontop of the GL context)

    cudaSetDevice(0);


    //=====================================OPENGL SETUP=========================================
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glClearColor(0.5f, 0.5f, 0.5f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    SDL_GL_SwapWindow(state_sdl.window);

    //=====================================CAMERA SETUP=========================================

    auto camera = Camera(windowWidth, windowHeight);

    rotateCamera(camera, TO_RAD(-45.f));
    pitchCamera(camera,  TO_RAD(-45.f));

    updateCamera(camera);

    //=====================================SHADER SETUP=========================================

    GLuint monoColourShader = compileMonoColourShaderProgram();
    if (monoColourShader == 0){
        std::cout<<"Mono Colour Shader failed to compile."<<std::endl;
        return 1;
    }


    //=====================================OPENGL DATA SETUP====================================

    //Create center of world grid plain
    std::vector<float> gridPlaneVertexData;
    //generateTile(gridPlaneVertexData);
    //generateLine(gridPlaneVertexData);
    generatePlaneVertexData(gridPlaneVertexData, 1, 6, 6);
    PlaneGLData gridPlane(&gridPlaneVertexData, &monoColourShader);
    initPlaneVAO(gridPlane);

    spag::Geometry test_geom(&gridPlaneVertexData, &monoColourShader);


    bool quit = false;
    while(!quit)
    {
        SDL_Event event;
        if (SDL_PollEvent(&event) != 0){
            switch (event.type){
                case SDL_QUIT:
                    quit = true;
                    break;

                case SDL_KEYDOWN:
                    switch(event.key.keysym.sym){
                    case SDLK_LEFT:
                        moveCamera(camera, Camera::ORBIT_LEFT);
                        std::cout<<"SDLK_LEFT"<<std::endl;
                        break;
                    case SDLK_RIGHT:
                        moveCamera(camera, Camera::ORBIT_RIGHT);
                        std::cout<<"SDLK_RIGHT"<<std::endl;
                        break;
                    case SDLK_UP:
                        moveCamera(camera, Camera::ORBIT_UP);
                        std::cout<<"SDLK_UP"<<std::endl;
                        break;
                    case SDLK_DOWN:
                        moveCamera(camera, Camera::ORBIT_DOWN);
                        std::cout<<"SDLK_DOWN"<<std::endl;
                        break;
                    case SDLK_e:{
                        spag::Geometry test_geom2(&gridPlaneVertexData, &monoColourShader);
                        
                    }
                        break;

                    case SDLK_w:
                        //spag::translateGeom(test_geom, float3{0,0.0001,0});
                        spag::addToBufferVertex<<<1, static_cast<int>((float)test_geom.buffer.d_bufferSize/3.f)>>>
                            (float3{0,0.01,0}, test_geom.buffer.d_pBuffer, test_geom.buffer.d_bufferSize);
                        break;
                    }
                    break;

                case SDL_MOUSEWHEEL:
                    if(event.wheel.y < 0)       // scroll up
                        moveCamera(camera, Camera::ZOOM_IN);
                    else if(event.wheel.y > 0)  // scroll down
                        moveCamera(camera, Camera::ZOOM_OUT);
                    break;
            }
        }

        glClearColor(.5f, 0.5f, 0.5f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        updateCamera(camera);
        Matrix4f cameraVP = camera.projMat * camera.viewMat;

        // Main stuff
        //updatePlaneVAO(gridPlane);
        //drawPlane(gridPlane, cameraVP);

        drawGeom(test_geom, cameraVP);

        glColor3f(0.0f,0.0f,1.0f); //blue color
        glBegin(GL_POINTS); //starts drawing of points
            glVertex3f(1.0f,1.0f,0.0f);//upper-right corner
            glVertex3f(-1.0f,-1.0f,0.0f);//lower-left corner
        glEnd();//end drawing of points

        SDL_GL_SwapWindow(state_sdl.window);
    }
    teardown(state_sdl);
}