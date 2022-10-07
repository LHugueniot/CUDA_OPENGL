
#include "Utils/ImGUI.h"
#include "Utils/Mesh.h"

#include "Camera.h"
#include "CuGlBuffer.cuh"
#include "CuGlGeometry.cuh"
#include "GLFWState.h"
#include "Geometry.cuh"
#include "GeometryViewer.cuh"
#include "GlGeometry.cuh"
#include "LoadGeometry.cuh"
#include "MacGrid.cuh"
#include "MonoColourGLShader.h"
#include "PlaneGLData.h"
#include "PositionBasedDynamics/PBDSolver.cuh"

// struct CUDA_GL_state{
//     int deviceCount;
//     cudaDeviceProp deviceProp;
//     std::vector<int> dev;
// };

// CUDA_GL_state initCUDAGLContext(){
//     CUDA_GL_state state;
//     cudaGetDeviceCount(&state.deviceCount);
//     for (int i=0; i<state.deviceCount; i++) {
//         cudaGetDeviceProperties(&state.deviceProp, state.dev[i]);
//     }
//     // Set Gl device to 1st device (in case of multiple GPUs ?)
//     cudaGLSetGLDevice(state.dev[0]);
//     return state;
// }

void ImGuiHelloWorld(bool showDemoWindow, ImVec4 &clearColor)
{
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and
                                   // append into it.

    ImGui::Text("This is some useful text."); // Display some text (you can use a
                                              // format strings too)
    ImGui::Checkbox(
        "Demo Window",
        &showDemoWindow); // Edit bools storing our window open/close state
    ImGui::Checkbox("Another Window", &showDemoWindow);

    ImGui::SliderFloat("float", &f, 0.0f,
                       1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::ColorEdit3("clear color",
                      (float *)&clearColor); // Edit 3 floats representing a color

    if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets
                                 // return true when edited/activated)
        counter++;
    ImGui::SameLine();
    ImGui::Text("counter = %d", counter);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();
}

int main(int argv, char **args)
{

    fprintf(stdout, "Start of main.\n");

    //=====================================GLEW/GLFW
    // SETUP=======================================

    int glMajorVersion = 4;
    int glMinorVersion = 6;
    // TODO: Change to 450
    const char *glslVersion = "#version 330";

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

    //=====================================IMGUI
    // SETUP==========================================

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable
    // Keyboard Controls io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; //
    // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(mainWindow->m_glfwWindow, true);
    ImGui_ImplOpenGL3_Init(glslVersion);

    bool showDemoWindow = true;

    //=====================================CUDA
    // SETUP==============================================
    // Initialize CUDA context (on top of the GL context)

    fprintf(stdout, "CUDA SETUP\n");

    // cudaSetDevice(0);
    // cudaGLSetGLDevice(0);

    //=====================================OPENGL
    // SETUP============================================

    fprintf(stdout, "OPENGL SETUP\n");

    ImVec4 clearColor = {0.5f, 0.5f, 0.5f, 1.f};

    glClearColor(clearColor.x * clearColor.w, clearColor.y * clearColor.w,
                 clearColor.z * clearColor.w, clearColor.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    // glDepthFunc(GL_LESS);

    // glfwSwapBuffers(mainWindow->m_glfwWindow);

    // During init, enable debug output
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(GLDebugMessageCallback, 0);

    //=====================================CAMERA
    // SETUP============================================

    fprintf(stdout, "CAMERA SETUP\n");

    auto camera = Camera(mainWindow->m_windowWidth, mainWindow->m_windowHeight);

    yawCamera(camera, TO_RAD(-45.f));
    pitchCamera(camera, TO_RAD(-45.f));

    updateCamera(camera);

    using KEY_ID = int;
    // std::map<KEY_ID, Camera::Actions> cameraKeyToAction = {
    //     {GLFW_KEY_UP, Camera::ORBIT_UP},
    //     {GLFW_KEY_LEFT, Camera::ORBIT_LEFT},
    //     {GLFW_KEY_RIGHT, Camera::ORBIT_RIGHT},
    //     {GLFW_KEY_DOWN, Camera::ORBIT_DOWN}
    // };
    std::map<KEY_ID, Camera::Actions> cameraKeyToAction = {
        {GLFW_KEY_W, Camera::ORBIT_UP}, {GLFW_KEY_A, Camera::ORBIT_LEFT}, {GLFW_KEY_D, Camera::ORBIT_RIGHT}, {GLFW_KEY_S, Camera::ORBIT_DOWN},

        {GLFW_KEY_UP, Camera::MOVE_X_P},
        {GLFW_KEY_LEFT, Camera::MOVE_Z_M},
        {GLFW_KEY_RIGHT, Camera::MOVE_Z_P},
        {GLFW_KEY_DOWN, Camera::MOVE_X_M}};

    //=====================================SHADER
    // SETUP============================================

    fprintf(stdout, "SHADER SETUP\n");

    GLuint monoColourShader = compileMonoColourShaderProgram();
    if (monoColourShader == 0)
    {
        fprintf(stderr, "Shader setup failed.\n");
        return EXIT_FAILURE;
    }

    checkGLError();

    //=====================================SCENE DATA
    // LOAD=========================================

    const aiScene *sceneCache = nullptr;

    std::filesystem::path assetFile(__FILE__);

    // assetFile = std::filesystem::absolute(
    //     assetFile.parent_path() / ".." / "assets" / "PantherBoss" / "PAN.obj");
    assetFile = std::filesystem::absolute(assetFile.parent_path() / ".." /
                                          "assets" / "cube_simple.obj");
    std::cout << assetFile << std::endl;

    std::vector<const aiMesh *> meshes = loadAiMeshes(assetFile, &sceneCache);

    //=====================================MESH DATA
    // SETUP=========================================

    fprintf(stdout, "MESH SETUP\n");

    // Create center of world grid plain
    std::vector<float> gridPlaneVertexData;
    ei::Vector3f cubeGridOrigin = {0.f, 0.f, 0.f};

    // generateSquare(gridPlaneVertexData,
    //                     ei::Vector3f(0.f, 0.f, 0.f),
    //                     1.f,
    //                     {Dim::X, Dim::Z});
    generateSquarePlane(gridPlaneVertexData, ei::Vector3f(0.f, 0.f, 0.f), 1.f,
                        {Dim::X, Dim::Z}, ei::Vector2ui(10, 10));
    // generateCubeGrid(gridPlaneVertexData,
    //                  cubeGridOrigin,
    //                  1.f,
    //                  ei::Vector3ui(10, 10, 10));
    float cubeGridTranslate[4] = {
        cubeGridOrigin[0],
        cubeGridOrigin[1],
        cubeGridOrigin[2],
        0,
    };

    // PlaneGLData gridPlane(&gridPlaneVertexData, &monoColourShader);

    // initPlaneVAO(gridPlane);

    CuGlGeometry gridPlaneCu(&gridPlaneVertexData, &monoColourShader);

    CuGlBufferSetter<float> vertexBufferSetter;
    CuGlBufferSetter<uint, GL_ELEMENT_ARRAY_BUFFER> indexBufferSetter;
    std::vector<std::pair<std::string, Geometry *>> nameToGeometry =
        initGeometryFromAiMeshes<Geometry>(meshes, vertexBufferSetter, {},
                                           indexBufferSetter);

    Geometry &cudaPanther = *(nameToGeometry[0].second);

    GeometryViewer patherViewer;

    initGeometryViewer(patherViewer, vertexBufferSetter.m_glBufferId,
                       vertexBufferSetter.m_nElements,
                       indexBufferSetter.m_glBufferId,
                       indexBufferSetter.m_nElements, &monoColourShader);

    std::vector<float> retrievedVertexData;
    retrievedVertexData.resize(vertexBufferSetter.m_data.size());
    cutilSafeCall(cudaMemcpy(&retrievedVertexData.data()[0],
                             cudaPanther.d_vertexPositionBufferData,
                             vertexBufferSetter.m_data.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));

    assert(retrievedVertexData == vertexBufferSetter.m_data);
    //=====================================MAIN
    // LOOP===============================================

    while (!shouldQuit(glfwState))
    {
        for (auto window : glfwState.m_windows)
        {
            glfwMakeContextCurrent(window->m_glfwWindow);

            glClearColor(clearColor.x * clearColor.w, clearColor.y * clearColor.w,
                         clearColor.z * clearColor.w, clearColor.w);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Handle interaction
            // TODO: handle better than.. this?
            glfwPollEvents();

            for (auto [key, action] : cameraKeyToAction)
            {
                int state = glfwGetKey(window->m_glfwWindow, key);
                if (state == GLFW_PRESS)
                {
                    // std::cout<<"Key pressed: "<<key<<std::endl;
                    moveCamera(camera, action);
                }
            }

            if (window->m_yScroll > 0)
            {
                moveCamera(camera, Camera::ZOOM_IN);
                window->m_yScroll = 0;
            }
            else if (window->m_yScroll < 0)
            {
                moveCamera(camera, Camera::ZOOM_OUT);
                window->m_yScroll = 0;
            }

            /*
            //translateGeom();
            */

            // Start ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            bool showDemoWindow = true;
            {
                ImGui::ShowDemoWindow(&showDemoWindow);
            }

            ImGuiHelloWorld(showDemoWindow, clearColor);

            {

                bool show_another_window = true;
                ImGui::Begin("Grid Translation", &show_another_window);
                ImGui::InputFloat3("input float3", cubeGridTranslate);
                ImGui::SameLine();
                if (ImGui::Button("Translate"))
                    translateGeom(gridPlaneCu,
                                  {cubeGridTranslate[0], cubeGridTranslate[1],
                                   cubeGridTranslate[2]});
                ImGui::End();
            }

            ImGui::Render();

            // Draw geometry
            // updatePlaneVBO(gridPlane);

            updateCamera(camera);
            // ei::Matrix4f cameraVP = camera.projMat.transpose() *
            // camera.viewMat.transpose();
            ei::Matrix4f cameraVP = camera.projMat * camera.viewMat;
            // cameraVP.transpose();
            // std::cout<<cameraVP<<std::endl;
            // updatePlaneVAO(gridPlane);
            // drawPlane(gridPlane, cameraVP);

            /*
                Geometry& cudaPanther = *(nameToGeometry[0].second);

                GeometryViewer patherViewer;

                initGeometryViewer(patherViewer,
                                   vertexBufferSetter.m_glBufferId,
                                   vertexBufferSetter.m_nElements,
                                   indexBufferSetter.m_glBufferId,
                                   indexBufferSetter.m_nElements,
                                   &monoColourShader);

                        vertexBufferSetter.m_resourceObj;
                        patherViewer.
            */

            // Map buffer object
            // cutilSafeCall(cudaGraphicsMapResources(1,
            // &vertexBufferSetter.m_resourceObj, 0));
            //
            // size_t size;
            // cutilSafeCall(cudaGraphicsResourceGetMappedPointer(
            //    reinterpret_cast<void**>(cudaPanther.d_vertexPositionBufferData),
            //    &size,
            //    m_resourceObj));
            //
            // cudaPanther.
            drawGeometryViewer(patherViewer, cameraVP);
            checkGLError();

            // drawGeom(gridPlaneCu, cameraVP);
            // checkGLError();

            // Overlay imgui stuff
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window->m_glfwWindow);
        }
    }

    teardown(glfwState);
    return EXIT_SUCCESS;
}