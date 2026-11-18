cmake_minimum_required(VERSION 3.24)
project(crispembed_spacy LANGUAGES C CXX)

# Closed standalone target: do not configure the parent OCR/document project.
get_filename_component(CRISPEMBED_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Static ggml dependencies" FORCE)
set(GGML_CPU ON CACHE BOOL "CPU backend" FORCE)
# CRISPEMBED_SPACY_SGEMM (default ON since 18 November 2026) keeps ggml's portable
# tiled SGEMM (llamafile tinyBLAS) for the default ggml engine. It changes only
# the F32/F16 mul_mat kernels; the scalar reference path
# (CRISPEMBED_SPACY_GGML=0) never uses it. -DCRISPEMBED_SPACY_SGEMM=OFF rebuilds
# the previous ggml kernel set.
option(CRISPEMBED_SPACY_SGEMM "Enable ggml tiled SGEMM for the spaCy ggml engine" ON)
set(_spacy_disabled_ggml NATIVE BACKEND_DL CPU_ALL_VARIANTS CPU_HBM CPU_KLEIDIAI CPU_REPACK
    OPENMP ACCELERATE BLAS CUDA HIP MUSA CANN VULKAN METAL
    SYCL RPC OPENCL OPENVINO WEBGPU ZDNN ZENDNN VIRTGPU
    VIRTGPU_BACKEND ET HEXAGON BUILD_TESTS BUILD_EXAMPLES)
if(NOT CRISPEMBED_SPACY_SGEMM)
    list(APPEND _spacy_disabled_ggml LLAMAFILE)
else()
    set(GGML_LLAMAFILE ON CACHE BOOL "Tiled SGEMM for the default spaCy ggml engine" FORCE)
endif()
foreach(flag IN LISTS _spacy_disabled_ggml)
    set(GGML_${flag} OFF CACHE BOOL "Disabled in the closed spaCy CPU target" FORCE)
endforeach()
set(CRISPEMBED_NATIVE OFF CACHE BOOL "Portable numeric development build" FORCE)
if(NOT EXISTS "${CRISPEMBED_ROOT}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "Initialize the pinned ggml submodule before configuring tools/spacy")
endif()
add_subdirectory("${CRISPEMBED_ROOT}/ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
find_package(Threads REQUIRED)
if(WIN32)
    # <windows.h> (src/core/gguf_loader.cpp) must not define min/max macros.
    add_compile_definitions(NOMINMAX)
endif()
find_path(SPACY_PCRE2_INCLUDE_DIR NAMES pcre2.h REQUIRED)
find_library(SPACY_PCRE2_LIBRARY NAMES pcre2-8 REQUIRED)
# A static PCRE2 on Windows must be consumed with PCRE2_STATIC (no dllimport).
# Other platforms ignore the define; their builds are unchanged.
option(SPACY_PCRE2_STATIC "SPACY_PCRE2_LIBRARY is a static PCRE2 (Windows needs PCRE2_STATIC)" OFF)

# CRISPEMBED_SPACY_STATIC (default OFF) builds a static library for targets
# that must link the runtime into a signed app (iOS). Desktop/Android capsules
# keep the shared library.
option(CRISPEMBED_SPACY_STATIC "Build crispembed-spacy as a static library" OFF)
if(CRISPEMBED_SPACY_STATIC)
    set(_spacy_library_kind STATIC)
else()
    set(_spacy_library_kind SHARED)
endif()
add_library(crispembed-spacy ${_spacy_library_kind}
    "${CRISPEMBED_ROOT}/src/spacy_pipeline.cpp"
    "${CRISPEMBED_ROOT}/src/spacy_numeric.cpp"
    "${CRISPEMBED_ROOT}/src/spacy_ggml.cpp"
    "${CRISPEMBED_ROOT}/src/spacy_ner.cpp"
    "${CRISPEMBED_ROOT}/src/spacy_v2.cpp"
    "${CRISPEMBED_ROOT}/src/spacy_lexical.cpp"
    "${CRISPEMBED_ROOT}/src/spacy_decode.cpp"
    "${CRISPEMBED_ROOT}/src/core/gguf_loader.cpp")
target_include_directories(crispembed-spacy PUBLIC "${CRISPEMBED_ROOT}/src")
target_include_directories(crispembed-spacy PRIVATE "${SPACY_PCRE2_INCLUDE_DIR}")
target_link_libraries(crispembed-spacy PRIVATE ggml ggml-cpu Threads::Threads "${SPACY_PCRE2_LIBRARY}")
# CRISPEMBED_SPACY_TRF (default OFF): spaCy transformer pipelines (e.g.
# fr_dep_news_trf). Links the reference SentencePiece C++ library, the same
# implementation spaCy's curated-tokenizers wraps, from a pinned release.
option(CRISPEMBED_SPACY_TRF "Build spaCy transformer pipeline support (SentencePiece)" OFF)
if(CRISPEMBED_SPACY_TRF)
    include(FetchContent)
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
    set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(sentencepiece
        URL https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz
        URL_HASH SHA256=9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86)
    FetchContent_GetProperties(sentencepiece)
    if(NOT sentencepiece_POPULATED)
        FetchContent_Populate(sentencepiece)
        # SentencePiece 0.2.0 declares cmake_minimum_required(3.1); CMake 4 needs a floor.
        set(_spacy_policy_minimum "${CMAKE_POLICY_VERSION_MINIMUM}")
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
        add_subdirectory(${sentencepiece_SOURCE_DIR} ${sentencepiece_BINARY_DIR} EXCLUDE_FROM_ALL)
        set(CMAKE_POLICY_VERSION_MINIMUM "${_spacy_policy_minimum}")
    endif()
    target_sources(crispembed-spacy PRIVATE "${CRISPEMBED_ROOT}/src/spacy_transformer.cpp")
    target_include_directories(crispembed-spacy PRIVATE "${sentencepiece_SOURCE_DIR}/src")
    target_link_libraries(crispembed-spacy PRIVATE sentencepiece-static)
    target_compile_definitions(crispembed-spacy PRIVATE CRISPEMBED_SPACY_TRF)
    install(FILES "${sentencepiece_SOURCE_DIR}/LICENSE" DESTINATION share/licenses/crispembed-spacy
            RENAME SentencePiece-LICENSE)
endif()
if(CRISPEMBED_SPACY_STATIC)
    target_compile_definitions(crispembed-spacy PRIVATE CRISPEMBED_SPACY_BUILD)
else()
    target_compile_definitions(crispembed-spacy PRIVATE CRISPEMBED_SPACY_BUILD PUBLIC CRISPEMBED_SPACY_SHARED)
endif()
if(SPACY_PCRE2_STATIC)
    target_compile_definitions(crispembed-spacy PRIVATE PCRE2_STATIC)
endif()
if(MSVC)
    target_compile_options(crispembed-spacy PRIVATE /W4 /fp:strict /utf-8)
else()
    target_compile_options(crispembed-spacy PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test-spacy-pipeline "${CRISPEMBED_ROOT}/tests/test_spacy_pipeline.cpp")
    target_link_libraries(test-spacy-pipeline PRIVATE crispembed-spacy ggml ggml-cpu Threads::Threads)
    add_test(NAME spacy-pipeline-contract COMMAND test-spacy-pipeline)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-tmp")
    set_tests_properties(spacy-pipeline-contract PROPERTIES
        ENVIRONMENT "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/test-tmp;TEMP=${CMAKE_CURRENT_BINARY_DIR}/test-tmp")
    if(EXISTS "${CRISPEMBED_ROOT}/tests/test_spacy_numeric.cpp")
        add_executable(test-spacy-numeric
            "${CRISPEMBED_ROOT}/tests/test_spacy_numeric.cpp"
            "${CRISPEMBED_ROOT}/src/spacy_numeric.cpp")
        target_include_directories(test-spacy-numeric PRIVATE "${CRISPEMBED_ROOT}/src")
        if(MSVC)
            target_compile_options(test-spacy-numeric PRIVATE /fp:strict /utf-8)
        else()
            target_compile_options(test-spacy-numeric PRIVATE -fno-fast-math -ffp-contract=off)
        endif()
        add_test(NAME spacy-numeric-contract COMMAND test-spacy-numeric)
    endif()
    # Stage-level native vs original-spaCy diff (crispembed_diff.h reference archives
    # from tools/spacy/dump_spacy_reference.py). Developer tool; not installed.
    add_executable(crispembed-spacy-diff "${CMAKE_CURRENT_LIST_DIR}/spacy_diff.cpp")
    target_include_directories(crispembed-spacy-diff PRIVATE "${CRISPEMBED_ROOT}/src")
    target_link_libraries(crispembed-spacy-diff PRIVATE crispembed-spacy)
    foreach(component lexical decode)
        if(EXISTS "${CRISPEMBED_ROOT}/tests/test_spacy_${component}.cpp")
            add_executable(test-spacy-${component}
                "${CRISPEMBED_ROOT}/tests/test_spacy_${component}.cpp"
                "${CRISPEMBED_ROOT}/src/spacy_${component}.cpp")
            target_include_directories(test-spacy-${component} PRIVATE "${CRISPEMBED_ROOT}/src")
            if(component STREQUAL "lexical")
                target_include_directories(test-spacy-${component} PRIVATE "${SPACY_PCRE2_INCLUDE_DIR}")
                target_link_libraries(test-spacy-${component} PRIVATE "${SPACY_PCRE2_LIBRARY}")
                if(SPACY_PCRE2_STATIC)
                    target_compile_definitions(test-spacy-${component} PRIVATE PCRE2_STATIC)
                endif()
            endif()
            add_test(NAME spacy-${component}-contract COMMAND test-spacy-${component})
        endif()
    endforeach()
endif()

# C-ABI-only analysis runner for hosts without Python (simulators/emulators).
add_executable(spacy-smoke "${CMAKE_CURRENT_LIST_DIR}/spacy_smoke.cpp")
target_link_libraries(spacy-smoke PRIVATE crispembed-spacy)

install(TARGETS crispembed-spacy LIBRARY DESTINATION lib RUNTIME DESTINATION bin ARCHIVE DESTINATION lib)
install(FILES "${CRISPEMBED_ROOT}/src/spacy_pipeline.h" DESTINATION include)
install(FILES "${CRISPEMBED_ROOT}/LICENSE"
        DESTINATION share/licenses/crispembed-spacy RENAME CrispEmbed-LICENSE)
install(FILES "${CRISPEMBED_ROOT}/ggml/LICENSE"
        DESTINATION share/licenses/crispembed-spacy RENAME ggml-LICENSE)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/THIRD_PARTY_NOTICES"
        DESTINATION share/licenses/crispembed-spacy)
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/notices/"
        DESTINATION share/licenses/crispembed-spacy/notices)
