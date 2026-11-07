cmake_minimum_required(VERSION 3.16)
project(physical_f16_shared_link LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT DEFINED CRISPEMBED_SOURCE OR NOT DEFINED GGML_SOURCE)
    message(FATAL_ERROR "Pinned source roots are required")
endif()
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${GGML_SOURCE}" ggml)
find_package(Threads REQUIRED)
# Actual shared attention object and its direct decoder closure, not a fake stub
# for attention_cpu. Source-membership checks bind this closure to main SDK.
add_library(physical_shared_attention STATIC
    "${CRISPEMBED_SOURCE}/src/qwen35_attention.cpp"
    "${CRISPEMBED_SOURCE}/src/qwen35_linear.cpp"
    "${CRISPEMBED_SOURCE}/src/qwen35_delta.cpp"
    "${CRISPEMBED_SOURCE}/src/qwen35_profile.cpp"
    "${CRISPEMBED_SOURCE}/src/lighton4b_physical_f16_attention.cpp")
target_include_directories(physical_shared_attention PUBLIC "${CRISPEMBED_SOURCE}/src")
target_link_libraries(physical_shared_attention PUBLIC ggml ggml-base ggml-cpu Threads::Threads)
add_executable(physical-f16-shared-link "${CRISPEMBED_SOURCE}/tests/test_lighton4b_main_f16_link.cpp")
target_link_libraries(physical-f16-shared-link PRIVATE physical_shared_attention)
if(EMSCRIPTEN)
    target_link_options(physical-f16-shared-link PRIVATE "-sENVIRONMENT=node" "-sEXIT_RUNTIME=1" "-sALLOW_MEMORY_GROWTH=1")
endif()
