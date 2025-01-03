# 设置最低 CMake 版本
cmake_minimum_required(VERSION 3.10)
project(MyProject)
# 定义项目名称和语言
project(MyProject LANGUAGES CXX CUDA)
# 设置 C++ 标准（例如 C++17）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDA 11.8 REQUIRED)
#find_package(PCL REQUIRED)
#find_package(Eigen3 REQUIRED)
# include_directories(${PCL_INCLUDE_DIRS})
# include_directories(${EIGEN3_INCLUDE_DIR})
# 链接 PCL 的库
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    # Optional: Specify the arch of your CUDA hardware here
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_52,code=sm_52 --expt-relaxed-constexpr -Wno-deprecated-declarations) # -Wno-deprecated-declarations
ENDIF ()
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR})
include_directories(
    ${PROJECT_INCLUDE_DIR}
)

# 添加可执行文件
add_executable(main main.cu)
#target_link_libraries(main ${PCL_LIBRARIES})