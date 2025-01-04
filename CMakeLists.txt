cmake_minimum_required(VERSION 3.10)

# 项目名称
project(VoxelHash LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA 11.8 REQUIRED)

# 添加包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# 添加源文件
set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/main.cu
    ${PROJECT_SOURCE_DIR}/src/VoxelHash.cu
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
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})