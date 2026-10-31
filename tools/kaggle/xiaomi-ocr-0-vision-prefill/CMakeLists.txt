# Test-only standalone target list for the composed-vision prefill audit.
# Deliberately NOT the repository root CMakeLists.txt: the shared build file is
# left untouched, and this target compiles only the sources the audit needs.
cmake_minimum_required(VERSION 3.20)
project(crispembed_vision_prefill_audit LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(GGML_CUDA "Enable CUDA GPU acceleration" OFF)
option(GGML_METAL "Enable Metal support" OFF)
option(GGML_VULKAN "Enable Vulkan support" OFF)
option(GGML_BLAS "Enable BLAS acceleration" OFF)
option(GGML_OPENMP "Use OpenMP" OFF)
option(GGML_LLAMAFILE "Enable llamafile SGEMM" ON)
# The hosted CPU compiles and runs this binary, so native dispatch is the
# intended target rather than a portability hazard.
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
add_subdirectory(ggml EXCLUDE_FROM_ALL)

add_executable(test-xiaomi-vision-prefill
    tests/test_xiaomi_vision_prefill.cpp
    src/core/gguf_loader.cpp
    src/qwen35_vision.cpp
    src/qwen35_linear.cpp
    src/qwen35_norm.cpp
    src/qwen35_profile.cpp
    # linear_core_cpu lives in the delta unit, not qwen35_linear.cpp.
    src/qwen35_delta.cpp
    src/core/gguf_loader.cpp)
target_include_directories(test-xiaomi-vision-prefill PRIVATE src src/core ${CMAKE_CURRENT_BINARY_DIR}/ggml/include)
target_compile_options(test-xiaomi-vision-prefill PRIVATE -Wall -Wextra)
find_package(Threads REQUIRED)
target_link_libraries(test-xiaomi-vision-prefill PRIVATE ggml ggml-base ggml-cpu Threads::Threads)
