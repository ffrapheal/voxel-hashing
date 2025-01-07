cmake_minimum_required(VERSION 3.10)
project(VoxelHash LANGUAGES CXX CUDA)

SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

ADD_COMPILE_OPTIONS(-std=c++14)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

find_package(Eigen3 REQUIRED)

set(PCL_ROOT /usr/lib/x86_64-linux-gnu)
set(PCL_INCLUDE_DIRS /usr/include/pcl-1.10/pcl)

find_package(PCL 1.10 REQUIRED PATHS ${PCL_ROOT} NO_DEFAULT_PATH)

include_directories(${PCL_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS 
    -O3
    --expt-relaxed-constexpr
    -gencode arch=compute_89,code=sm_89
    -G
    --disable-warnings
    -Wno-deprecated-declarations
    -dc
)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

set(source_files src/VoxelHash.cu src/main.cu)

cuda_add_library(cuda_impl ${source_files})

target_link_libraries(cuda_impl ${PCL_LIBRARIES} yaml-cpp)

add_executable(${PROJECT_NAME} src/main.cu)
target_link_libraries(${PROJECT_NAME} cuda_impl yaml-cpp)

