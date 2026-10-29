cmake_minimum_required(VERSION 3.18)
project(crispembed_minicpm_graph_controls LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CRISPEMBED_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
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

# Exact fixed-ISA helper from root CMakeLists.txt; no independent ISA policy.
function(crispembed_apply_arch_flags target)
    if(EMSCRIPTEN OR MSVC OR CMAKE_CROSSCOMPILING)
        return()
    endif()
    if(CRISPEMBED_NATIVE)
        target_compile_options(${target} PRIVATE -march=native)
        return()
    endif()
    # The GGML_AVX*/FMA/F16C options exist on every host (ggml declares them
    # unconditionally), so gate on the target architecture before emitting x86
    # flags — `-mavx2` on aarch64 is a hard compile error.
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^([Xx]86_64|[Aa][Mm][Dd]64|i[3-6]86)$")
        return()
    endif()
    set(_arch_flags "")
    if(GGML_AVX512)
        list(APPEND _arch_flags -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)
    endif()
    if(GGML_AVX2)
        list(APPEND _arch_flags -mavx2)
    endif()
    if(GGML_FMA)
        list(APPEND _arch_flags -mfma)
    endif()
    if(GGML_F16C)
        list(APPEND _arch_flags -mf16c)
    endif()
    if(_arch_flags)
        target_compile_options(${target} PRIVATE ${_arch_flags})
    endif()
endfunction()
foreach(target minicpm-graph-controls minicpm-graph-probe minicpm-graph-diff)
    crispembed_apply_arch_flags(${target})
endforeach()
