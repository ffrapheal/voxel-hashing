# 设置最低 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 定义项目名称和语言
project(MyProject LANGUAGES CXX CUDA)

# 设置 C++ 标准（例如 C++17）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDA 10.0 REQUIRED)
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    # Optional: Specify the arch of your CUDA hardware here
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_52,code=sm_52 --expt-relaxed-constexpr -Wno-deprecated-declarations) # -Wno-deprecated-declarations
ENDIF ()
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(
    ${PROJECT_INCLUDE_DIR}
)
message("${CUDA_INCLUDE_DIRS}")
# 添加可执行文件
add_executable(main main.cu)