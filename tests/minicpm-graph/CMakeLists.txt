cmake_minimum_required(VERSION 3.18)
project(crispembed_minicpm_graph_controls LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(CRISPEMBED_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${CRISPEMBED_ROOT}/ggml" ggml)
find_package(Threads REQUIRED)
add_executable(minicpm-graph-controls controls.cpp
    "${CRISPEMBED_ROOT}/src/qwen35_graph_decoder.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_linear.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_delta.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_profile.cpp")
target_include_directories(minicpm-graph-controls PRIVATE "${CRISPEMBED_ROOT}/src")
target_link_libraries(minicpm-graph-controls PRIVATE ggml ggml-cpu ggml-base Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(minicpm-graph-controls PRIVATE -ffp-contract=off)
endif()
add_executable(minicpm-graph-probe probe.cpp
    "${CRISPEMBED_ROOT}/src/qwen35_graph_decoder.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_decoder.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_attention.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_linear.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_delta.cpp"
    "${CRISPEMBED_ROOT}/src/qwen35_profile.cpp"
    "${CRISPEMBED_ROOT}/src/core/gguf_loader.cpp")
target_include_directories(minicpm-graph-probe PRIVATE "${CRISPEMBED_ROOT}/src")
target_compile_definitions(minicpm-graph-probe PRIVATE CRISPEMBED_MINICPM_GRAPH_AUDIT=1)
target_link_libraries(minicpm-graph-probe PRIVATE ggml ggml-cpu ggml-base Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(minicpm-graph-probe PRIVATE -ffp-contract=off)
endif()
add_executable(minicpm-graph-diff diff.cpp)
target_include_directories(minicpm-graph-diff PRIVATE "${CRISPEMBED_ROOT}/src")
