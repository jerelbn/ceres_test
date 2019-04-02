cmake_minimum_required(VERSION 2.8.3)
project(ceres_test)

# default to release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS "-std=c++11 -mavx2 -mfma -fopenmp")

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)

include_directories(include
  lib/common_cpp/include
  lib/geometry/include
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
)

add_library(common_cpp lib/common_cpp/src/common.cpp)

add_executable(rosenbrock src/rosenbrock.cpp)
target_link_libraries(rosenbrock ${CERES_LIBRARIES})

add_executable(rosenbrock_eigen src/rosenbrock_eigen.cpp)
target_link_libraries(rosenbrock_eigen ${CERES_LIBRARIES})

add_executable(optimize_rot src/optimize_r_only.cpp)
target_link_libraries(optimize_rot common_cpp ${CERES_LIBRARIES})

add_executable(optimize_rt src/optimize_rt.cpp)
target_link_libraries(optimize_rt common_cpp ${CERES_LIBRARIES})

add_executable(quat_tests src/quat_tests.cpp)
target_link_libraries(quat_tests common_cpp ${CERES_LIBRARIES})

add_executable(vo_1d src/vo_1d.cpp)
target_link_libraries(vo_1d ${CERES_LIBRARIES})

add_executable(vo_2d src/vo_2d.cpp)
target_link_libraries(vo_2d ${CERES_LIBRARIES})

add_executable(jacobians src/jacobians.cpp)
target_link_libraries(jacobians ${CERES_LIBRARIES})

add_executable(state_jacobian src/state_jacobian.cpp)
target_link_libraries(state_jacobian common_cpp ${CERES_LIBRARIES})

add_executable(manifold_jacobians src/manifold_jacobians.cpp)
target_link_libraries(manifold_jacobians common_cpp ${CERES_LIBRARIES})

add_executable(lm_opt src/lm_opt.cpp)
target_link_libraries(lm_opt ${CERES_LIBRARIES})

add_executable(lc_vi_mhe src/lc_vi_mhe/lc_vi_mhe_test.cpp src/lc_vi_mhe/lc_vi_mhe.cpp src/lc_vi_mhe/true_state.cpp)
target_link_libraries(lc_vi_mhe ${CERES_LIBRARIES} -lstdc++fs)
