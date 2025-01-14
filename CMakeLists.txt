cmake_minimum_required(VERSION 3.10)

# 项目名称
project(VoxelHash)
project(VoxelHash LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0074 NEW)

# 查找CUDA
find_package(CUDA REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
cmake_policy(SET CMP0074 NEW)

# 添加PCL库的具体路径
link_directories(${PCL_ROOT})
# 添加包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
# 添加源文件
set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/main.cu
    ${PROJECT_SOURCE_DIR}/src/VoxelHash.cu
    ${PROJECT_SOURCE_DIR}/src/CUDAMarchingCubesHashSDF.cpp
    ${PROJECT_SOURCE_DIR}/src/CUDAMarchingCubesSDF.cu
)

# 添加可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 设置CUDA编译选项
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 添加编译选项
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-deprecated-declarations>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Wno-deprecated-declarations>
)

# 链接CUDA库
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${CUDA_LIBRARIES} yaml-cpp)
