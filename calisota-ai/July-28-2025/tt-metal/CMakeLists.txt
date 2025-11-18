cmake_minimum_required(VERSION 3.18)
project(DynamicCortex VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(BUILD_TT_METAL "Build TT-Metal backend" ON)
option(BUILD_CUDA "Build CUDA backend" ON)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(ENABLE_PROFILING "Enable performance profiling" OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --extended-lambda")

# CUDA architecture
if(BUILD_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4080 (Ada Lovelace)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/tt-metal/core)

# Find packages
if(BUILD_TT_METAL)
    find_path(TT_METAL_INCLUDE_DIR tt_metal/host_api.hpp
              PATHS $ENV{TT_METAL_HOME}/include
              NO_DEFAULT_PATH)

    if(TT_METAL_INCLUDE_DIR)
        message(STATUS "Found TT-Metal: ${TT_METAL_INCLUDE_DIR}")
        include_directories(${TT_METAL_INCLUDE_DIR})
        add_definitions(-DHAS_TT_METAL)
    else()
        message(WARNING "TT-Metal SDK not found. Set TT_METAL_HOME environment variable.")
        set(BUILD_TT_METAL OFF)
    endif()
endif()

if(BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_FOUND)
        message(STATUS "Found CUDA: ${CUDAToolkit_VERSION}")
    endif()
endif()

# Source files
set(CORE_SOURCES
    src/core/dynamic_cortex_impl.cpp
    src/core/tensor.cpp
    src/core/utils.cpp
)

# TT-Metal backend
if(BUILD_TT_METAL)
    set(TT_METAL_SOURCES
        src/tt-metal/kernels/v1_encode.cpp
        src/tt-metal/kernels/lm_modulate.cpp
        src/tt-metal/kernels/feedback_transform.cpp
    )

    add_library(dynamic_cortex_ttmetal STATIC ${TT_METAL_SOURCES})
    target_compile_definitions(dynamic_cortex_ttmetal PRIVATE HAS_TT_METAL)

    if(TT_METAL_INCLUDE_DIR)
        target_include_directories(dynamic_cortex_ttmetal PRIVATE ${TT_METAL_INCLUDE_DIR})
    endif()
endif()

# CUDA backend
if(BUILD_CUDA)
    set(CUDA_SOURCES
        src/cuda/kernels/v1_encode.cu
        src/cuda/kernels/lm_modulate.cu
        src/cuda/kernels/feedback_transform.cu
    )

    add_library(dynamic_cortex_cuda STATIC ${CUDA_SOURCES})
    target_compile_options(dynamic_cortex_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --ptxas-options=-v
        >
    )
    target_link_libraries(dynamic_cortex_cuda
        CUDA::cudart
        CUDA::cublas
    )
endif()

# Main library (stub implementations when SDK not available)
# add_library(dynamic_cortex SHARED ${CORE_SOURCES})

# Examples
if(BUILD_EXAMPLES)
    if(BUILD_TT_METAL)
        add_executable(dynamic_cortex_demo_ttmetal
            src/tt-metal/examples/dynamic_cortex_demo.cpp
        )
        target_link_libraries(dynamic_cortex_demo_ttmetal
            dynamic_cortex_ttmetal
        )

        add_executable(visual_task_simulator_ttmetal
            src/tt-metal/examples/visual_task_simulator.cpp
        )
        target_link_libraries(visual_task_simulator_ttmetal
            dynamic_cortex_ttmetal
        )
    endif()

    if(BUILD_CUDA)
        add_executable(dynamic_cortex_demo_cuda
            src/cuda/examples/dynamic_cortex_demo.cu
        )
        target_link_libraries(dynamic_cortex_demo_cuda
            dynamic_cortex_cuda
        )
        set_target_properties(dynamic_cortex_demo_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_stream_config tests/test_stream_config.cpp)
    add_test(NAME StreamConfig COMMAND test_stream_config)

    add_executable(test_tensor_ops tests/test_tensor_ops.cpp)
    add_test(NAME TensorOps COMMAND test_tensor_ops)

    if(BUILD_TT_METAL)
        target_link_libraries(test_stream_config dynamic_cortex_ttmetal)
        target_link_libraries(test_tensor_ops dynamic_cortex_ttmetal)
    endif()
endif()

# Installation
install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")

if(BUILD_TT_METAL)
    install(TARGETS dynamic_cortex_ttmetal
            ARCHIVE DESTINATION lib)
endif()

if(BUILD_CUDA)
    install(TARGETS dynamic_cortex_cuda
            ARCHIVE DESTINATION lib)
endif()

# Configuration file
configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/DynamicCortexConfig.cmake.in
    ${CMAKE_BINARY_DIR}/DynamicCortexConfig.cmake
    @ONLY
)

install(FILES ${CMAKE_BINARY_DIR}/DynamicCortexConfig.cmake
        DESTINATION lib/cmake/DynamicCortex)

# Print configuration summary
message(STATUS "")
message(STATUS "Dynamic Cortex Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build TT-Metal backend: ${BUILD_TT_METAL}")
message(STATUS "  Build CUDA backend: ${BUILD_CUDA}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Enable profiling: ${ENABLE_PROFILING}")
message(STATUS "")
