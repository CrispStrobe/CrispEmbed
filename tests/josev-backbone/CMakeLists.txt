cmake_minimum_required(VERSION 3.20)
project(crispembed_josev_backbone LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
get_filename_component(CRISPEMBED_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
add_subdirectory("${CRISPEMBED_ROOT}/ggml" ggml-build)
find_package(Threads REQUIRED)
add_library(josev-backbone STATIC
    "${CRISPEMBED_ROOT}/src/josev2b_backbone.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_decoder.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_attention.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_linear.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_delta.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_norm.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_positions.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_profile.cpp"
    "${CRISPEMBED_ROOT}/src/core/gguf_loader.cpp")
target_include_directories(josev-backbone PUBLIC "${CRISPEMBED_ROOT}/src")
target_link_libraries(josev-backbone PUBLIC ggml ggml-cpu Threads::Threads)
add_executable(test-josev-backbone metadata.cpp)
target_link_libraries(test-josev-backbone PRIVATE josev-backbone)
add_executable(josev-backbone-probe probe.cpp)
target_link_libraries(josev-backbone-probe PRIVATE josev-backbone)
# Actual header-only Ref comparator; no root CMake or runtime dependency.
add_executable(josev-backbone-diff diff.cpp)
target_include_directories(josev-backbone-diff PRIVATE "${CRISPEMBED_ROOT}/src")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(josev-backbone PRIVATE -ffp-contract=off)
  target_compile_options(test-josev-backbone PRIVATE -ffp-contract=off)
endif()
