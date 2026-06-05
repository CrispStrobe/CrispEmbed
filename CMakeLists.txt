cmake_minimum_required(VERSION 3.14)

# Read project version from the VERSION file (single source of truth shared
# with release.yml, crispembed-sys/Cargo.toml, etc.). Bumping the file is
# the release ritual.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" CRISPEMBED_VERSION_RAW)
string(STRIP "${CRISPEMBED_VERSION_RAW}" CRISPEMBED_VERSION_RAW)
project(crispembed VERSION ${CRISPEMBED_VERSION_RAW} LANGUAGES C CXX)

# SOVERSION 0: ABI is not yet declared stable. Bump to 1 when cutting v1.0.
set(CRISPEMBED_SOVERSION 0)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

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
    src/image_preprocess.cpp
    examples/cli/model_mgr.cpp
)
if(CRISPEMBED_HAS_CRISP_AUDIO)
    list(APPEND CRISPEMBED_SOURCES src/bidirlm_audio.cpp)
endif()
list(APPEND CRISPEMBED_SOURCES src/bidirlm_vision.cpp)
list(APPEND CRISPEMBED_SOURCES src/vit_embed.cpp)
list(APPEND CRISPEMBED_SOURCES src/cnn_embed.cpp)
list(APPEND CRISPEMBED_SOURCES src/face_align.cpp)
list(APPEND CRISPEMBED_SOURCES src/math_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/hmer_ocr.cpp)

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

# Installed binaries live in <prefix>/bin and need to find their .so/.dylib
# siblings in <prefix>/lib. The literal $ORIGIN must be single-quoted to
# survive CMake's processing.
if(APPLE)
    set(CRISPEMBED_BIN_INSTALL_RPATH "@loader_path/../lib")
elseif(UNIX)
    set(CRISPEMBED_BIN_INSTALL_RPATH "$ORIGIN/../lib")
else()
    set(CRISPEMBED_BIN_INSTALL_RPATH "")
endif()

add_executable(crispembed-cli
    examples/cli/main.cpp
)
target_include_directories(crispembed-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli
)
target_link_libraries(crispembed-cli PRIVATE crispembed)
set_target_properties(crispembed-cli PROPERTIES
    OUTPUT_NAME  crispembed
    INSTALL_RPATH "${CRISPEMBED_BIN_INSTALL_RPATH}"
)
install(TARGETS crispembed-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(crispembed-server
    examples/server/server.cpp
)
target_include_directories(crispembed-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/server
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli
)
target_link_libraries(crispembed-server PRIVATE crispembed)
set_target_properties(crispembed-server PROPERTIES
    INSTALL_RPATH "${CRISPEMBED_BIN_INSTALL_RPATH}"
)
install(TARGETS crispembed-server RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(crispembed-quantize
    tools/quantize.cpp
)
target_include_directories(crispembed-quantize PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_link_libraries(crispembed-quantize PRIVATE ggml ggml-base ggml-cpu)
set_target_properties(crispembed-quantize PROPERTIES
    OUTPUT_NAME  crispembed-quantize
    INSTALL_RPATH "${CRISPEMBED_BIN_INSTALL_RPATH}"
)
install(TARGETS crispembed-quantize RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(test-hmer tests/test_hmer.cpp)
target_link_libraries(test-hmer PRIVATE crispembed)

endif() # NOT iOS

# ---------------------------------------------------------------------------
# Shared library — the public, distributable face of CrispEmbed.
#
# Layout after `cmake --install build`:
#   <prefix>/lib/libcrispembed.so.${VERSION}     (real file)
#   <prefix>/lib/libcrispembed.so.0              (SONAME symlink)
#   <prefix>/lib/libcrispembed.so                (linker symlink)
#   <prefix>/lib/libggml*.so                     (ggml siblings, from ggml/)
#   <prefix>/include/crispembed.h                (PUBLIC_HEADER)
#   <prefix>/lib/cmake/crispembed/               (config + targets + version)
#   <prefix>/lib/pkgconfig/crispembed.pc
#
# RPATH is set to $ORIGIN (Linux) / @loader_path (macOS) so the .so finds its
# ggml siblings without LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
# ---------------------------------------------------------------------------
if(CRISPEMBED_BUILD_SHARED)
    add_library(crispembed-shared SHARED ${CRISPEMBED_SOURCES})

    # Use BUILD_INTERFACE/INSTALL_INTERFACE so the include path differs for
    # in-tree consumers (src/) vs find_package() consumers (<prefix>/include).
    target_include_directories(crispembed-shared PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ggml/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/examples/cli>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_definitions(crispembed-shared PRIVATE CRISPEMBED_BUILD)
    if(CRISPEMBED_HAS_CRISP_AUDIO)
        target_compile_definitions(crispembed-shared PRIVATE CRISPEMBED_HAS_CRISP_AUDIO)
    endif()
    # All ggml + crisp_audio + crispembed-core are STATIC libs baked into the
    # shared .so/.dylib. Mark them PRIVATE so they don't leak into the export
    # set — consumers of crispembed::crispembed (find_package) only see the
    # C ABI in crispembed.h, never these internal symbols.
    target_link_libraries(crispembed-shared PRIVATE
        crispembed-core ggml ggml-base ggml-cpu Threads::Threads
    )
    if(CRISPEMBED_HAS_CRISP_AUDIO)
        target_link_libraries(crispembed-shared PRIVATE crisp_audio)
    endif()

    set_target_properties(crispembed-shared PROPERTIES
        OUTPUT_NAME   crispembed
        VERSION       ${PROJECT_VERSION}
        SOVERSION     ${CRISPEMBED_SOVERSION}
        PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/crispembed.h"
        # Sibling-resolved RPATH so a release tarball is self-contained.
        # Build-tree RPATH preserved so dev iteration works without install.
        BUILD_WITH_INSTALL_RPATH FALSE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    if(APPLE)
        set_target_properties(crispembed-shared PROPERTIES
            INSTALL_RPATH "@loader_path")
    elseif(UNIX)
        set_target_properties(crispembed-shared PROPERTIES
            INSTALL_RPATH "$ORIGIN")
    endif()

    # ---------- install ---------------------------------------------------
    # We deliberately omit `EXPORT crispembed-targets` because the shared
    # lib's transitive (private) link to ggml-cpu — itself a SHARED library
    # — trips install(EXPORT)'s dependency-set check. Instead, the IMPORTED
    # target is hand-rolled in `cmake/crispembed-config.cmake.in` via
    # find_library(), which gives downstream `find_package(crispembed)` an
    # equivalent `crispembed::crispembed` target with no ggml leakage.
    install(TARGETS crispembed-shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}        # *.so / *.dylib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}        # *.lib (Windows import lib)
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}        # *.dll (Windows)
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # crispembed-config.cmake + crispembed-config-version.cmake
    set(CRISPEMBED_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
    set(CRISPEMBED_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR})
    set(CRISPEMBED_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR})
    set(CRISPEMBED_BUILD_COMMIT        "unknown")

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/crispembed-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/crispembed-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/crispembed
        PATH_VARS
            CRISPEMBED_INCLUDE_INSTALL_DIR
            CRISPEMBED_LIB_INSTALL_DIR
            CRISPEMBED_BIN_INSTALL_DIR
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/crispembed-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/crispembed-config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/crispembed-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/crispembed
    )

    # pkg-config for autotools / meson consumers.
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/crispembed.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/crispembed.pc
        @ONLY
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/crispembed.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )
endif()
