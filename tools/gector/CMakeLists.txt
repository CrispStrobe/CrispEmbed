cmake_minimum_required(VERSION 3.24)
project(crispembed_gector LANGUAGES C CXX)

# A deliberately closed standalone target. Do not add the parent CMake project:
# that target also builds OCR, document parsers and optional sibling packages.
get_filename_component(CRISPEMBED_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Only ggml's CPU implementation and its MIT SGEMM are admitted by this target.
# The generated compile_commands.json and actual linked-library inventory still
# belong in a platform qualification receipt; this is not a binary license claim.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build ggml dependencies statically" FORCE)
set(GGML_CPU ON CACHE BOOL "CPU backend" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "MIT SGEMM" FORCE)
foreach(flag NATIVE BACKEND_DL CPU_ALL_VARIANTS CPU_HBM CPU_KLEIDIAI
             OPENMP ACCELERATE BLAS CUDA HIP MUSA CANN VULKAN METAL SYCL RPC
             OPENCL OPENVINO WEBGPU ZDNN ZENDNN VIRTGPU VIRTGPU_BACKEND ET HEXAGON
             BUILD_TESTS BUILD_EXAMPLES)
    set(GGML_${flag} OFF CACHE BOOL "Disabled in the closed GECToR CPU target" FORCE)
endforeach()
set(CRISPEMBED_NATIVE OFF CACHE BOOL "Portable grammar build" FORCE)
if(NOT EXISTS "${CRISPEMBED_ROOT}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "Initialize the pinned ggml submodule before configuring tools/gector")
endif()
add_subdirectory("${CRISPEMBED_ROOT}/ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

add_library(crispembed-gector SHARED
    "${CRISPEMBED_ROOT}/src/gector.cpp"
    "${CRISPEMBED_ROOT}/src/core/gguf_loader.cpp")
target_include_directories(crispembed-gector PUBLIC "${CRISPEMBED_ROOT}/src")
target_link_libraries(crispembed-gector PRIVATE ggml ggml-cpu)
target_compile_definitions(crispembed-gector PRIVATE CRISPEMBED_GECTOR_BUILD PUBLIC CRISPEMBED_GECTOR_SHARED)
if(NOT MSVC)
    target_compile_options(crispembed-gector PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
endif()

add_executable(crispembed-gector-run main.cpp)
target_link_libraries(crispembed-gector-run PRIVATE crispembed-gector)

# Reuse the existing ggml quantizer without pulling any general runtime sources.
add_executable(crispembed-gector-quantize "${CRISPEMBED_ROOT}/tools/quantize.cpp")
target_include_directories(crispembed-gector-quantize PRIVATE "${CRISPEMBED_ROOT}/src")
target_link_libraries(crispembed-gector-quantize PRIVATE ggml ggml-cpu)

include(CTest)
if(BUILD_TESTING)
    add_executable(test-gector "${CRISPEMBED_ROOT}/tests/test_gector.cpp")
    target_link_libraries(test-gector PRIVATE crispembed-gector ggml ggml-cpu)
    add_test(NAME gector-contract COMMAND test-gector)
endif()

install(TARGETS crispembed-gector LIBRARY DESTINATION lib RUNTIME DESTINATION bin ARCHIVE DESTINATION lib)
install(FILES "${CRISPEMBED_ROOT}/src/gector.h" DESTINATION include)
install(FILES "${CRISPEMBED_ROOT}/LICENSE" "${CMAKE_CURRENT_LIST_DIR}/THIRD_PARTY_NOTICES"
        DESTINATION share/licenses/crispembed-gector)
