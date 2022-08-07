
#include "Utils/ImGUI.h"

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


void ImGuiHelloWorld(bool showDemoWindow, ImVec4& clearColor)
{
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

    ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    ImGui::Checkbox("Demo Window", &showDemoWindow);      // Edit bools storing our window open/close state
    ImGui::Checkbox("Another Window", &showDemoWindow);

    ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::ColorEdit3("clear color", (float*)&clearColor); // Edit 3 floats representing a color

    if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
        counter++;
    ImGui::SameLine();
    ImGui::Text("counter = %d", counter);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);
    ImGui::End();
}

int main(int argv, char** args)
{

    fprintf(stdout, "Start of main.\n");

    //=====================================GLEW/GLFW SETUP=======================================

    int glMajorVersion = 4;
    int glMinorVersion = 6;
    const char * glslVersion = "#version 330";


    fprintf(stdout, "GLEW/GLFW SETUP\n");

    // Initialize GLFW
    auto glfwState = setupGLFW(glMajorVersion, glMinorVersion);

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

    //=====================================IMGUI SETUP==========================================

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(mainWindow->m_glfwWindow, true);
    ImGui_ImplOpenGL3_Init(glslVersion);

    bool showDemoWindow = true;

    //=====================================CUDA SETUP===========================================
    // Initialize CUDA context (on top of the GL context)

    fprintf(stdout, "CUDA SETUP\n");

    //cudaSetDevice(0);
    //cudaGLSetGLDevice(0);

    //=====================================OPENGL SETUP=========================================
    
    fprintf(stdout, "OPENGL SETUP\n");

    ImVec4 clearColor = {0.5f, 0.5f, 0.5f, 1.f};

    glClearColor(clearColor.x * clearColor.w,
                 clearColor.y * clearColor.w,
                 clearColor.z * clearColor.w,
                 clearColor.w);
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
    Geometry gridPlaneCu(&gridPlaneVertexData, &monoColourShader);
    /*
    */
    glfwMakeContextCurrent(mainWindow->m_glfwWindow);
    //glViewport(0, 0, mainWindow->m_windowWidth, mainWindow->m_windowHeight);

    while(!shouldQuit(glfwState))
    {
        //fprintf(stdout, "Drawing for window: %s\n", mainWindow->m_windowTitle.c_str());


        for (auto window : glfwState.m_windows)
        {
            glfwMakeContextCurrent(window->m_glfwWindow);

            glClearColor(clearColor.x * clearColor.w,
                         clearColor.y * clearColor.w,
                         clearColor.z * clearColor.w,
                         clearColor.w);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Handle interaction
            // TODO: handle better than.. this?
            glfwPollEvents();

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

            // Start ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGuiHelloWorld(showDemoWindow, clearColor);

            ImGui::Render();

            // Draw geometry
            //updatePlaneVBO(gridPlane);

            updateCamera(camera);
            //ei::Matrix4f cameraVP = camera.projMat.transpose() * camera.viewMat.transpose();
            ei::Matrix4f cameraVP = camera.projMat * camera.viewMat;
            //cameraVP.transpose();
            std::cout<<cameraVP<<std::endl;
            //updatePlaneVAO(gridPlane);
            //drawPlane(gridPlane, cameraVP);
            drawGeom(gridPlaneCu, cameraVP);
            check_gl_error();

            // Overlay imgui stuff
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window->m_glfwWindow);
        }
    }

    teardown(glfwState);
    return EXIT_SUCCESS;
}