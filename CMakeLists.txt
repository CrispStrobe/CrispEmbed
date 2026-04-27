cmake_minimum_required(VERSION 3.14)
project(crispembed LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static archives are linked into libcrispembed.so on Linux, so everything
# needs position-independent code (otherwise ld errors with R_X86_64_PC32
# relocations against glibc symbols).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Suppress MSVC deprecation warnings for standard C functions
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(GGML_BLAS        "Enable BLAS acceleration (OpenBLAS/MKL)" OFF)
option(GGML_CUDA        "Enable CUDA GPU acceleration" OFF)
option(GGML_VULKAN      "Enable Vulkan GPU acceleration" OFF)
option(GGML_METAL       "Enable Metal GPU acceleration (macOS)" OFF)
option(GGML_LLAMAFILE   "Enable llamafile optimized SGEMM" ON)
option(CRISPEMBED_BUILD_SHARED "Build shared library for Python" ON)

# BLAS vendor: Generic, OpenBLAS, Intel10_64lp (MKL), Apple, etc.
if(NOT DEFINED GGML_BLAS_VENDOR)
    if(APPLE)
        set(GGML_BLAS_VENDOR "Apple" CACHE STRING "BLAS vendor")
    else()
        set(GGML_BLAS_VENDOR "Generic" CACHE STRING "BLAS vendor")
    endif()
endif()

# ---------------------------------------------------------------------------
# ggml (submodule)
# ---------------------------------------------------------------------------
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "ggml submodule not found. Please run:\n"
        "  git submodule update --init --recursive\n"
        "Or clone with: git clone --recursive https://github.com/CrispStrobe/CrispEmbed")
endif()

add_subdirectory(ggml)

# ---------------------------------------------------------------------------
# crisp_audio — shared audio-encoder library, lives in CrispASR. The path is
# overridable via -DCRISP_AUDIO_DIR=... ; default is the sibling-repo layout
# the team uses on dev machines (~/code/CrispEmbed alongside ~/code/CrispASR).
# When the directory is missing we silently skip — audio paths will fail at
# runtime with a clear error from crispembed_encode_audio() instead.
# ---------------------------------------------------------------------------
set(CRISP_AUDIO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CrispASR/crisp_audio"
    CACHE PATH "Path to CrispASR's crisp_audio subdirectory")
set(CRISPEMBED_HAS_CRISP_AUDIO OFF)
if(EXISTS "${CRISP_AUDIO_DIR}/CMakeLists.txt")
    message(STATUS "crispembed: linking crisp_audio from ${CRISP_AUDIO_DIR}")
    set(CRISPEMBED_HAS_CRISP_AUDIO ON)
else()
    message(STATUS "crispembed: crisp_audio not found at ${CRISP_AUDIO_DIR} — audio paths disabled")
endif()

# ---------------------------------------------------------------------------
# Core GGUF loader library
# ---------------------------------------------------------------------------
add_library(crispembed-core STATIC
    src/core/gguf_loader.cpp
    src/core/mel.cpp
)
target_include_directories(crispembed-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_link_libraries(crispembed-core PUBLIC ggml ggml-base)

# Pull crisp_audio in AFTER crispembed-core is defined: crisp_audio's
# CMakeLists picks crispembed-core as its dependency target.
if(CRISPEMBED_HAS_CRISP_AUDIO)
    add_subdirectory(${CRISP_AUDIO_DIR}
                     ${CMAKE_BINARY_DIR}/crisp_audio)
endif()

# ---------------------------------------------------------------------------
# CrispEmbed library
# ---------------------------------------------------------------------------
set(CRISPEMBED_SOURCES
    src/crispembed.cpp
    src/decoder_embed.cpp
    src/tokenizer.cpp
    src/tokenizer_spm.cpp
    src/tokenizer_bpe.cpp
    examples/cli/model_mgr.cpp
)
if(CRISPEMBED_HAS_CRISP_AUDIO)
    list(APPEND CRISPEMBED_SOURCES src/bidirlm_audio.cpp)
endif()

add_library(crispembed STATIC ${CRISPEMBED_SOURCES})
if(CRISPEMBED_HAS_CRISP_AUDIO)
    target_compile_definitions(crispembed PRIVATE CRISPEMBED_HAS_CRISP_AUDIO)
endif()
target_include_directories(crispembed PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli
)
# Avoid name clash with the DLL import lib (both would be crispembed.lib on Windows)
set_target_properties(crispembed PROPERTIES ARCHIVE_OUTPUT_NAME crispembed-static)

find_package(Threads REQUIRED)
target_link_libraries(crispembed PUBLIC
    crispembed-core
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(CRISPEMBED_HAS_CRISP_AUDIO)
    target_link_libraries(crispembed PUBLIC crisp_audio)
endif()

# ---------------------------------------------------------------------------
# CLI binaries (skip on iOS — no standalone executables)
# ---------------------------------------------------------------------------
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")

add_executable(crispembed-cli
    examples/cli/main.cpp
)
target_include_directories(crispembed-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli
)
target_link_libraries(crispembed-cli PRIVATE crispembed)
set_target_properties(crispembed-cli PROPERTIES OUTPUT_NAME crispembed)
install(TARGETS crispembed-cli RUNTIME)

add_executable(crispembed-server
    examples/server/server.cpp
)
target_include_directories(crispembed-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/server
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli
)
target_link_libraries(crispembed-server PRIVATE crispembed)
install(TARGETS crispembed-server RUNTIME)

add_executable(crispembed-quantize
    tools/quantize.cpp
)
target_include_directories(crispembed-quantize PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_link_libraries(crispembed-quantize PRIVATE ggml ggml-base ggml-cpu)
set_target_properties(crispembed-quantize PROPERTIES OUTPUT_NAME crispembed-quantize)
install(TARGETS crispembed-quantize RUNTIME)

endif() # NOT iOS

# ---------------------------------------------------------------------------
# Shared library for Python bindings
# ---------------------------------------------------------------------------
if(CRISPEMBED_BUILD_SHARED)
    add_library(crispembed-shared SHARED ${CRISPEMBED_SOURCES})
    target_include_directories(crispembed-shared PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli
    )
    target_compile_definitions(crispembed-shared PRIVATE CRISPEMBED_BUILD)
    if(CRISPEMBED_HAS_CRISP_AUDIO)
        target_compile_definitions(crispembed-shared PRIVATE CRISPEMBED_HAS_CRISP_AUDIO)
    endif()
    target_link_libraries(crispembed-shared PUBLIC
        crispembed-core ggml ggml-base ggml-cpu Threads::Threads
    )
    if(CRISPEMBED_HAS_CRISP_AUDIO)
        target_link_libraries(crispembed-shared PUBLIC crisp_audio)
    endif()
    set_target_properties(crispembed-shared PROPERTIES OUTPUT_NAME crispembed)
endif()
