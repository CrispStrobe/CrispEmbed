cmake_minimum_required(VERSION 3.20)
project(crispembed_josev_composed LANGUAGES C CXX)
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
set(COMPOSED_SOURCES
    "${CRISPEMBED_ROOT}/src/josev2b_backbone.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_decoder.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_attention.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_linear.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_delta.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_norm.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_positions.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_profile.cpp"
    "${CRISPEMBED_ROOT}/src/core/gguf_loader.cpp"
    "${CRISPEMBED_ROOT}/src/josev2b_composed.cpp"
    "${CRISPEMBED_ROOT}/src/josev2b_head.cpp"
    "${CRISPEMBED_ROOT}/src/josev2b_text_frontend.cpp"
    "${CRISPEMBED_ROOT}/src/josev2b_answer_serializer.cpp"
    "${CRISPEMBED_ROOT}/src/core/shortest_double.c")

foreach(variant plain observed)
  add_library(josev-composed-${variant} STATIC ${COMPOSED_SOURCES})
  target_include_directories(josev-composed-${variant} PUBLIC "${CRISPEMBED_ROOT}/src")
  target_link_libraries(josev-composed-${variant} PUBLIC ggml ggml-cpu Threads::Threads)
  add_executable(josev-composed-${variant}-probe probe.cpp)
  target_link_libraries(josev-composed-${variant}-probe PRIVATE josev-composed-${variant})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(josev-composed-${variant} PRIVATE -ffp-contract=off)
  endif()
endforeach()
target_compile_definitions(josev-composed-observed PRIVATE CRISPEMBED_JOSEV_COMPOSED_TESTING CRISPEMBED_JOSEV_HEAD_TESTING)
target_compile_definitions(josev-composed-observed-probe PRIVATE CRISPEMBED_JOSEV_COMPOSED_TESTING)

add_executable(josev-composed-diff diff.cpp)
target_include_directories(josev-composed-diff PRIVATE "${CRISPEMBED_ROOT}/src")
