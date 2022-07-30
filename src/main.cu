#include "Camera.h"
#include "GLFWState.h"
#include "MacGrid.cuh"
#include "PlaneGLData.h"
#include "MonoColourGLShader.h"
#include "Geometry.cuh"
#include "CuGlBuffer.cuh"

#include "SimpleTriangle.h"


// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtx/string_cast.hpp>

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

void print_glm_mat4(glm::mat4 & mat)
{
    for (int r=0; r<4; r++){
        for (int c=0; c<4; c++)
        {
            fprintf(stdout, "%7.1f", mat[r][c]);
        }
        fprintf(stdout, "\n");
    }
}


int main1(int argv, char** args)
{
    float radians_fov = 45.0f;
    float windowWidth = 1920.f;
    float windowHeight = 1080.f;
    float fov = 45.f;
    float far = 100.0f;
    float near = 0.1f;


	glm::mat4 glm_projection = glm::perspective(
        glm::radians(radians_fov), windowWidth / windowHeight, near, far
    );
    glm::mat4 glm_view = glm::lookAt(
        glm::vec3(0, 0, 30), // Camera is at (4,3,-3), in World Space
        glm::vec3(0, 0, 0), // and looks at the origin
        glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
	);

    glm::mat4 glm_PV = glm_projection * glm_view;

    ei::Matrix4f ei_projection;
    ei::Matrix4f ei_view;

    setProjMat(ei_projection, windowWidth, windowHeight, TO_RAD(fov), far, near);
    setLookAt(ei_view, {0, 0, 30}, {0, 0, 0}, {0, 1, 0});

    //ei::Matrix4f ei_PV = ei_projection.transpose() * ei_view.transpose();
    ei::Matrix4f ei_PV = ei_projection * ei_view;

    assert(TO_RAD(fov) == glm::radians(radians_fov));

    std::cout << "PROJECTION MATRICES" << std::endl;
    std::cout << ei_projection << std::endl;
    print_glm_mat4(glm_projection);
    
    std::cout << "VIEW MATRICES" << std::endl;
    std::cout << ei_view << std::endl;
    print_glm_mat4(glm_view);
    
    std::cout << "PROJECTION VIEW MATRICES" << std::endl;
    std::cout << ei_PV.transpose() << std::endl;
    print_glm_mat4(glm_PV);
    //std::cout << glm::to_string(glm_projection) << std::endl;
    

    for (int i=0; i<16; i++)
    {
        auto* p = &glm_projection[0][0];
        assert(p[i] == ei_projection.data()[i]);
        auto* v = &glm_view[0][0];
        assert(v[i] == ei_view.data()[i]);
        auto* pv = &glm_PV[0][0];
        assert(pv[i] == ei_PV.data()[i]);
    }

    return 0;

    for (int r=0; r<4; r++)
        for (int c=0; c<4; c++)
        {
            assert(glm_projection[r][c] == ei_projection(r, c));
            assert(glm_view[r][c] == ei_view(r, c));
            assert(glm_PV[r][c] == ei_PV(r, c));
        }

    return 0;
}

int main(int argv, char** args)
{

    fprintf(stdout, "Start of main.\n");

    //=====================================GLEW/GLFW SETUP=======================================

    fprintf(stdout, "GLEW/GLFW SETUP\n");

    // Initialize GLFW
    auto glfwState = setupGLFW(4, 6);

    // Create a GLFW window
    auto mainWindow = createWindow("Main Window");
    auto windowAddedResult = addWindow(glfwState, mainWindow);
	glfwMakeContextCurrent(mainWindow->m_glfwWindow);

    fprintf(stdout, "Adding window result: %i\n", windowAddedResult);
    fprintf(stdout, "m_windows size: %i\n", (int)glfwState.m_windows.size());

    if (glewInit() != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW.\n");
        return EXIT_FAILURE;
    }

    //=====================================CUDA SETUP===========================================
    // Initialize CUDA context (on top of the GL context)

    fprintf(stdout, "CUDA SETUP\n");

    //cudaSetDevice(0);
    //cudaGLSetGLDevice(0);

    //=====================================OPENGL SETUP=========================================
    
    fprintf(stdout, "OPENGL SETUP\n");

    glClearColor(0.5f, 0.5f, 0.5f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    //glDepthFunc(GL_LESS);

    //glfwSwapBuffers(mainWindow->m_glfwWindow);

    //=====================================CAMERA SETUP=========================================

    fprintf(stdout, "CAMERA SETUP\n");

    auto camera = Camera(mainWindow->m_windowWidth, mainWindow->m_windowHeight);

    yawCamera(camera, TO_RAD(-45.f));
    pitchCamera(camera,  TO_RAD(-45.f));

    updateCamera(camera);

    using KEY_ID=int;
    //std::map<KEY_ID, Camera::Actions> cameraKeyToAction = {
    //    {GLFW_KEY_UP, Camera::ORBIT_UP},
    //    {GLFW_KEY_LEFT, Camera::ORBIT_LEFT},
    //    {GLFW_KEY_RIGHT, Camera::ORBIT_RIGHT},
    //    {GLFW_KEY_DOWN, Camera::ORBIT_DOWN}
    //};
    std::map<KEY_ID, Camera::Actions> cameraKeyToAction = {
        {GLFW_KEY_W, Camera::ORBIT_UP},
        {GLFW_KEY_A, Camera::ORBIT_LEFT},
        {GLFW_KEY_D, Camera::ORBIT_RIGHT},
        {GLFW_KEY_S, Camera::ORBIT_DOWN},
    
        {GLFW_KEY_UP, Camera::MOVE_X_P},
        {GLFW_KEY_LEFT, Camera::MOVE_Z_M},
        {GLFW_KEY_RIGHT, Camera::MOVE_Z_P},
        {GLFW_KEY_DOWN, Camera::MOVE_X_M}
    };

    //=====================================SHADER SETUP=========================================

    fprintf(stdout, "SHADER SETUP\n");
    
    GLuint monoColourShader = compileMonoColourShaderProgram();
    if (monoColourShader == 0)
    {
        fprintf(stderr, "Shader setup failed.\n");
        return EXIT_FAILURE;
    }

    check_gl_error();
    //=====================================MESH DATA SETUP====================================

    fprintf(stdout, "MESH SETUP\n");
    
    //Create center of world grid plain
    std::vector<float> gridPlaneVertexData;
    //generateTile(gridPlaneVertexData);
    //generateLine(gridPlaneVertexData);
    generatePlaneVertexData(gridPlaneVertexData, 1, 6, 6);
    PlaneGLData gridPlane(&gridPlaneVertexData, &monoColourShader);
    initPlaneVAO(gridPlane);
    /*
    Geometry test_geom(&gridPlaneVertexData, &monoColourShader);
    */
    //SimpleTriangle triangle;

    glfwMakeContextCurrent(mainWindow->m_glfwWindow);
    //glViewport(0, 0, mainWindow->m_windowWidth, mainWindow->m_windowHeight);

    while(!shouldQuit(glfwState))
    {
        //fprintf(stdout, "Drawing for window: %s\n", mainWindow->m_windowTitle.c_str());


        for (auto window : glfwState.m_windows)
        {
            glfwMakeContextCurrent(window->m_glfwWindow);

            glClearColor(.5f, .5f, .5f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            for (auto [key, action] : cameraKeyToAction)
            {
                int state = glfwGetKey(window->m_glfwWindow, key);
                if (state == GLFW_PRESS)
                {
                    //std::cout<<"Key pressed: "<<key<<std::endl;
                    moveCamera(camera, action);
                }
            }
            if (window->m_yScroll > 0)
            {
                moveCamera(camera, Camera::ZOOM_IN);
                window->m_yScroll = 0;
                std::cout<<"ZOOM_IN"<<std::endl;
            }
            else if(window->m_yScroll < 0)
            {
                moveCamera(camera, Camera::ZOOM_OUT);
                std::cout<<"ZOOM_OUT"<<std::endl;
                window->m_yScroll = 0;
            }
            
            updatePlaneVBO(gridPlane);

            updateCamera(camera);
            //ei::Matrix4f cameraVP = camera.projMat.transpose() * camera.viewMat.transpose();
            ei::Matrix4f cameraVP = camera.projMat * camera.viewMat;
            //cameraVP.transpose();
            std::cout<<cameraVP<<std::endl;
            //triangle.draw(cameraVP);
            /*
            updateCamera(camera);
            ei::Matrix4f cameraVP = camera.projMat * camera.viewMat;
            // Main stuff
            */
            //updatePlaneVAO(gridPlane);
            drawPlane(gridPlane, cameraVP);
            //drawGeom(test_geom, cameraVP);
            check_gl_error();

            glfwSwapBuffers(window->m_glfwWindow);
            glfwPollEvents();
        }
    }

    teardown(glfwState);
    return EXIT_SUCCESS;
}