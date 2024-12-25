cmake_minimum_required(VERSION 3.10)

project(Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDA REQUIRED)

IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    # Optional: Specify the arch of your CUDA hardware here
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -expt-relaxed-constexpr -Wno-deprecated-declarations) # -Wno-deprecated-declarations
    # -gencode arch=compute_52,code=sm_52 
ENDIF ()

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

include_directories(
    ${PROJECT_INCLUDE_DIR}
)

message("${CUDA_INCLUDE_DIRS}")

add_executable(main main.cu)