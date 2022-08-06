#include "gtest/gtest.h"
#include <sstream>
#include <vector>
#include <string>
#include "Camera.h"

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>


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

TEST(EigenTests, ProjectionAndViewTest)
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

    ei_utils::setProjMat(ei_projection, windowWidth, windowHeight, TO_RAD(fov), far, near);
    ei_utils::setLookAt(ei_view, {0, 0, 30}, {0, 0, 0}, {0, 1, 0});

    //ei::Matrix4f ei_PV = ei_projection.transpose() * ei_view.transpose();
    ei::Matrix4f ei_PV = ei_projection * ei_view;

    ASSERT_EQ(TO_RAD(fov), glm::radians(radians_fov));

    for (int i=0; i<16; i++)
    {
        EXPECT_NEAR(glm::value_ptr(glm_projection)[i], ei_projection.data()[i], 0.1);
        EXPECT_NEAR(glm::value_ptr(glm_view)[i], ei_view.data()[i], 0.1);
        EXPECT_NEAR(glm::value_ptr(glm_PV)[i], ei_PV.data()[i], 0.1);
    }

//    std::cout << "PROJECTION MATRICES" << std::endl;
//    std::cout << ei_projection << std::endl;
//    print_glm_mat4(glm_projection);
//    
//    std::cout << "VIEW MATRICES" << std::endl;
//    std::cout << ei_view << std::endl;
//    print_glm_mat4(glm_view);
//    
//    std::cout << "PROJECTION VIEW MATRICES" << std::endl;
//    std::cout << ei_PV.transpose() << std::endl;
//    print_glm_mat4(glm_PV);
//    //std::cout << glm::to_string(glm_projection) << std::endl;
//    
//
//    for (int i=0; i<16; i++)
//    {
//        auto* p = &glm_projection[0][0];
//        assert(p[i] == ei_projection.data()[i]);
//        auto* v = &glm_view[0][0];
//        assert(v[i] == ei_view.data()[i]);
//        auto* pv = &glm_PV[0][0];
//        assert(pv[i] == ei_PV.data()[i]);
//    }
//
//    for (int r=0; r<4; r++)
//        for (int c=0; c<4; c++)
//        {
//            assert(glm_projection[r][c] == ei_projection(r, c));
//            assert(glm_view[r][c] == ei_view(r, c));
//            assert(glm_PV[r][c] == ei_PV(r, c));
//        }
}

TEST(testTest, test)
{
    EXPECT_EQ(true, true);
}