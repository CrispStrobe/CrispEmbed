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
    # MSVC's <cmath> only defines M_PI etc. with this. Define it globally so
    # source files don't each need a `#ifndef M_PI` guard (a recurring Windows
    # build break: text_sr/pan_sr/restormer/swinir_sr all hit it).
    add_definitions(-D_USE_MATH_DEFINES)
    # <windows.h> defines min/max as macros, which breaks every `std::min(a, b)`
    # in a translation unit that pulls it in transitively — MSVC reports it as
    # C2589 "'(': illegal token on right side of '::'" and then cascades into
    # bogus "function does not take N arguments" / "must have an initializer"
    # errors many lines later, so the real cause is nowhere near the first
    # message. Suppress the macros globally rather than per-call-site
    # `(std::min)(...)` parens, which only survive until the next author.
    add_definitions(-DNOMINMAX)
    # Without this, <windows.h> drags in the ancient <winsock.h>. Anything that
    # later includes <winsock2.h> — httplib.h does, so the whole server — then
    # fails with ~100 "'sockaddr': 'struct' type redefinition" errors in the
    # Windows SDK itself. It bites on include ORDER, so it is invisible until
    # someone adds a header: server.cpp includes core/temp_file.h (windows.h)
    # at line 38 and httplib.h (winsock2.h) at line 47.
    add_definitions(-DWIN32_LEAN_AND_MEAN)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(GGML_BLAS        "Enable BLAS acceleration (OpenBLAS/MKL)" OFF)
option(GGML_CUDA        "Enable CUDA GPU acceleration" OFF)
option(GGML_VULKAN      "Enable Vulkan GPU acceleration" OFF)
option(GGML_METAL       "Enable Metal GPU acceleration (macOS)" OFF)
# ggml declares option(GGML_METAL_EMBED_LIBRARY ... ${GGML_METAL}): its default
# follows GGML_METAL only at the FIRST configure of a build dir. A dir first
# configured without Metal caches EMBED_LIBRARY=OFF, and a later
# -DGGML_METAL=ON never revisits it — the metallib is then emitted to
# build/bin/ while ggml_metal_library_init probes next to argv[0], so Metal
# init fails and every run silently falls back to CPU behind a cache that
# still says GGML_METAL:BOOL=ON. Pin the embed to follow GGML_METAL on every
# configure so a stale cache cannot produce that state.
if (GGML_METAL)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "ggml: embed Metal library" FORCE)
endif()
option(GGML_LLAMAFILE   "Enable llamafile optimized SGEMM" ON)
option(GGML_OPENMP      "Enable OpenMP threading in ggml (OFF uses lighter pthreads pool)" OFF)
option(CRISPEMBED_BUILD_SHARED "Build shared library for Python" ON)

# CPU ISA baseline for CrispEmbed's own translation units.
#
# `-march=native` bakes in whatever the BUILD machine happens to support. That
# is right for a local build and wrong for anything shipped: issue #41 — a
# release runner that happened to have AVX-512 produced a Windows cpu artifact
# that died with "Illegal instruction" on a Raptor Lake i9 (AVX2, no AVX-512),
# while the previous release, built on an AVX2-only runner, ran fine. ggml has
# the same hazard via `GGML_NATIVE` (on MSVC its FindSIMD.cmake *runs* probe
# programs on the builder and turns on /arch:AVX512 if they succeed).
#
# So default CRISPEMBED_NATIVE to GGML_NATIVE: a release build passes
# `-DGGML_NATIVE=OFF` once and the WHOLE tree — ggml and CrispEmbed alike —
# compiles to a fixed, portable baseline. Setting only one of the two leaves
# half the binary tuned for the builder's CPU, which is how this bug ships.
if(DEFINED GGML_NATIVE)
    set(CRISPEMBED_NATIVE_DEFAULT ${GGML_NATIVE})
elseif(CMAKE_CROSSCOMPILING)
    # Mirrors ggml's own GGML_NATIVE_DEFAULT: you cannot probe the host when
    # the host is not the target.
    set(CRISPEMBED_NATIVE_DEFAULT OFF)
else()
    set(CRISPEMBED_NATIVE_DEFAULT ON)
endif()
option(CRISPEMBED_NATIVE       "Use -march=native for CPU SIMD (AVX2/NEON)" ${CRISPEMBED_NATIVE_DEFAULT})

# BLAS vendor: Generic, OpenBLAS, Intel10_64lp (MKL), Apple, etc.
if(NOT DEFINED GGML_BLAS_VENDOR)
    if(APPLE)
        set(GGML_BLAS_VENDOR "Apple" CACHE STRING "BLAS vendor")
    else()
        set(GGML_BLAS_VENDOR "Generic" CACHE STRING "BLAS vendor")
    endif()
endif()

# ---------------------------------------------------------------------------
# Emscripten: override FindThreads with a stub that creates a no-op
# Threads::Threads target. This avoids -pthread leaking into the build,
# which would require SharedArrayBuffer + COOP/COEP headers for deployment.
# The math OCR decoder runs single-threaded; ggml's thread pool compiles
# but degrades gracefully (pthread stubs in Emscripten libc).
# ---------------------------------------------------------------------------
option(CRISPEMBED_WASM_THREADS "Use real pthreads in WASM (requires COOP/COEP)" OFF)
if(EMSCRIPTEN)
    # Also used for threaded builds: CMake's real FindThreads fails under
    # Emscripten, and build-wasm.sh --threads already injects -pthread into
    # CMAKE_{C,CXX,EXE_LINKER}_FLAGS, so a no-op Threads::Threads suffices.
    list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
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
# CPU ISA flags for CrispEmbed's own targets
# ---------------------------------------------------------------------------
# `src/core/cpu_ops.h` guards its intrinsics on __AVX2__ / __FMA__ / __ARM_NEON,
# so the compiler needs to be told which ISA to target or those paths silently
# compile out to scalar code.
#
#   CRISPEMBED_NATIVE=ON  → -march=native. Fastest locally, NOT redistributable.
#   CRISPEMBED_NATIVE=OFF → mirror the fixed baseline ggml was configured with
#                           (GGML_AVX2 / GGML_FMA / GGML_F16C / GGML_AVX512),
#                           so the SIMD paths stay compiled in without
#                           over-shooting the ISA the ggml half targets.
#
# MSVC is excluded from both arms: it has no `-march`, and the crispembed
# targets have always been built there at the compiler default (SSE2). Raising
# that would be a separate, deliberate change — ggml-cpu carries the SIMD that
# matters on Windows.
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
# crisp_punc — shared punctuation library, lives in CrispASR.
# When present, replaces the local fireredpunc.cpp + pcs.cpp copies.
# ---------------------------------------------------------------------------
set(CRISP_PUNC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CrispASR/crisp_punc"
    CACHE PATH "Path to CrispASR's crisp_punc subdirectory")
set(CRISPEMBED_HAS_CRISP_PUNC OFF)
if(EXISTS "${CRISP_PUNC_DIR}/CMakeLists.txt")
    message(STATUS "crispembed: linking crisp_punc from ${CRISP_PUNC_DIR}")
    set(CRISPEMBED_HAS_CRISP_PUNC ON)
else()
    message(STATUS "crispembed: crisp_punc not found at ${CRISP_PUNC_DIR} — using local copies")
endif()

# ---------------------------------------------------------------------------
# crisp_lid — shared text LID library, lives in CrispASR.
# Enables OCR language routing (pick Tesseract model by detected language).
# ---------------------------------------------------------------------------
set(CRISP_LID_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CrispASR/crisp_lid"
    CACHE PATH "Path to CrispASR's crisp_lid subdirectory")
set(CRISPEMBED_HAS_CRISP_LID OFF)
if(EXISTS "${CRISP_LID_DIR}/CMakeLists.txt")
    message(STATUS "crispembed: linking crisp_lid from ${CRISP_LID_DIR}")
    set(CRISPEMBED_HAS_CRISP_LID ON)
else()
    message(STATUS "crispembed: crisp_lid not found at ${CRISP_LID_DIR} — LID disabled")
endif()

# ---------------------------------------------------------------------------
# crisp_truecase — shared truecasing library, lives in CrispASR.
# Post-OCR truecasing (German noun capitalization, etc.).
# ---------------------------------------------------------------------------
set(CRISP_TRUECASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../CrispASR/crisp_truecase"
    CACHE PATH "Path to CrispASR's crisp_truecase subdirectory")
set(CRISPEMBED_HAS_CRISP_TRUECASE OFF)
if(EXISTS "${CRISP_TRUECASE_DIR}/CMakeLists.txt")
    message(STATUS "crispembed: linking crisp_truecase from ${CRISP_TRUECASE_DIR}")
    set(CRISPEMBED_HAS_CRISP_TRUECASE ON)
else()
    message(STATUS "crispembed: crisp_truecase not found at ${CRISP_TRUECASE_DIR} — truecasing disabled")
endif()

# ---------------------------------------------------------------------------
# Core GGUF loader library
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CrispEmbedC2pa.cmake)

add_library(crispembed-core STATIC
    src/core/gguf_loader.cpp
    src/core/mel.cpp
    src/core/image_out.cpp
)
crispembed_enable_c2pa(crispembed-core)
target_include_directories(crispembed-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_link_libraries(crispembed-core PUBLIC ggml ggml-base)

# Pull crisp_audio in AFTER crispembed-core is defined: crisp_audio's
# CMakeLists picks crispembed-core as its dependency target.
if(CRISPEMBED_HAS_CRISP_PUNC)
    add_subdirectory(${CRISP_PUNC_DIR}
                     ${CMAKE_BINARY_DIR}/crisp_punc)
endif()
if(CRISPEMBED_HAS_CRISP_LID)
    add_subdirectory(${CRISP_LID_DIR}
                     ${CMAKE_BINARY_DIR}/crisp_lid)
endif()
if(CRISPEMBED_HAS_CRISP_TRUECASE)
    add_subdirectory(${CRISP_TRUECASE_DIR}
                     ${CMAKE_BINARY_DIR}/crisp_truecase)
endif()
if(CRISPEMBED_HAS_CRISP_AUDIO)
    add_subdirectory(${CRISP_AUDIO_DIR}
                     ${CMAKE_BINARY_DIR}/crisp_audio)
endif()

# ---------------------------------------------------------------------------
# CrispEmbed library
# ---------------------------------------------------------------------------
set(CRISPEMBED_SOURCES
    src/crispembed.cpp
    src/imatrix.cpp
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
list(APPEND CRISPEMBED_SOURCES src/clip_text_embed.cpp)
list(APPEND CRISPEMBED_SOURCES src/cnn_embed.cpp)
list(APPEND CRISPEMBED_SOURCES src/face_align.cpp)
list(APPEND CRISPEMBED_SOURCES src/math_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/smt_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/transcoda_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/tromr_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/flova_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/hmer_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/bttr_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/posformer_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/ppformulanet_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/ppformulanet_l_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/ppocrv6_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/ppocrv6_det.cpp)
list(APPEND CRISPEMBED_SOURCES src/pplcnet_orientation.cpp)
list(APPEND CRISPEMBED_SOURCES src/easyocr_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/easyocr_layout.cpp)
list(APPEND CRISPEMBED_SOURCES src/easyocr_pipeline.cpp)
list(APPEND CRISPEMBED_SOURCES src/easyocr_postprocess.cpp)
list(APPEND CRISPEMBED_SOURCES src/easyocr_craft.cpp)
list(APPEND CRISPEMBED_SOURCES src/ocr_detect.cpp)
list(APPEND CRISPEMBED_SOURCES src/ocr_pipeline.cpp)
list(APPEND CRISPEMBED_SOURCES src/ocr_crop.cpp)
list(APPEND CRISPEMBED_SOURCES src/ocr_pipeline_pool.cpp)
list(APPEND CRISPEMBED_SOURCES src/ocr_region_router.cpp)
list(APPEND CRISPEMBED_SOURCES src/ocr_orchestrator.cpp)
list(APPEND CRISPEMBED_SOURCES src/layout_detect.cpp)
list(APPEND CRISPEMBED_SOURCES src/surya_det.cpp)
list(APPEND CRISPEMBED_SOURCES src/mixtex_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/qwen2vl_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/internvl2_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/gliner_ner.cpp)
list(APPEND CRISPEMBED_SOURCES src/lfm2_embed.cpp)
list(APPEND CRISPEMBED_SOURCES src/scan_cleanup.cpp)
list(APPEND CRISPEMBED_SOURCES src/nafnet_denoise.cpp)
list(APPEND CRISPEMBED_SOURCES src/text_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/tbsrn_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/pan_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/hat_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/dat_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/deepseek_ocr2.cpp)
list(APPEND CRISPEMBED_SOURCES src/unlimited_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/restormer.cpp)
list(APPEND CRISPEMBED_SOURCES src/granite_vision_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/smoldocling_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/pix2struct.cpp)
list(APPEND CRISPEMBED_SOURCES src/parseq_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/glm_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/bert_ner.cpp)
list(APPEND CRISPEMBED_SOURCES src/got_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/lfm2_vl_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/lightonocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/tesseract_lstm.cpp)
list(APPEND CRISPEMBED_SOURCES src/tesseract_recoder.cpp src/tesseract_dawg_score.cpp)
list(APPEND CRISPEMBED_SOURCES src/tesseract_dawg.cpp src/tesseract_dawg_legacy.cpp)
list(APPEND CRISPEMBED_SOURCES src/tesseract_pageseg.cpp)
if(NOT CRISPEMBED_HAS_CRISP_PUNC)
    list(APPEND CRISPEMBED_SOURCES src/fireredpunc.cpp)
    list(APPEND CRISPEMBED_SOURCES src/pcs.cpp)
endif()
list(APPEND CRISPEMBED_SOURCES src/morph_fast.cpp)
list(APPEND CRISPEMBED_SOURCES src/cc_detect.cpp)
list(APPEND CRISPEMBED_SOURCES src/classical_preproc.cpp)
list(APPEND CRISPEMBED_SOURCES src/ocr_render.cpp)
list(APPEND CRISPEMBED_SOURCES src/dewarp.cpp)
list(APPEND CRISPEMBED_SOURCES src/kie_pipeline.cpp)
list(APPEND CRISPEMBED_SOURCES src/lilt_kie.cpp)
list(APPEND CRISPEMBED_SOURCES src/tps_warp.cpp)
list(APPEND CRISPEMBED_SOURCES src/tps_locnet.cpp)
list(APPEND CRISPEMBED_SOURCES src/safmn_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/adair.cpp)
list(APPEND CRISPEMBED_SOURCES src/instructir.cpp)
list(APPEND CRISPEMBED_SOURCES src/esrgan_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/swinir_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/scunet_denoise.cpp)
list(APPEND CRISPEMBED_SOURCES src/pdf_info.cpp)
list(APPEND CRISPEMBED_SOURCES src/table_parse.cpp)

add_library(crispembed STATIC ${CRISPEMBED_SOURCES})
# Version string for output provenance (hOCR ocr-system, ALTO
# softwareVersion, PDF XMP). An archived document should be traceable to a
# build, and naming the software without a version is not that.
target_compile_definitions(crispembed PRIVATE CRISPEMBED_VERSION_STR="${CRISPEMBED_VERSION_RAW}")
if(CRISPEMBED_HAS_CRISP_AUDIO)
    target_compile_definitions(crispembed PRIVATE CRISPEMBED_HAS_CRISP_AUDIO)
endif()
target_include_directories(crispembed PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli
)
# Enable SIMD intrinsics in cpu_ops.h (AVX2/FMA on x86, NEON on ARM)
crispembed_apply_arch_flags(crispembed)
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
if(GGML_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        target_link_libraries(crispembed PUBLIC ${BLAS_LIBRARIES})
    endif()
endif()
if(CRISPEMBED_HAS_CRISP_AUDIO)
    target_link_libraries(crispembed PUBLIC crisp_audio)
endif()
if(CRISPEMBED_HAS_CRISP_PUNC)
    target_link_libraries(crispembed PUBLIC crisp_punc)
endif()
if(CRISPEMBED_HAS_CRISP_LID)
    target_link_libraries(crispembed PUBLIC crisp_lid)
endif()
if(CRISPEMBED_HAS_CRISP_TRUECASE)
    target_link_libraries(crispembed PUBLIC crisp_truecase)
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

# Host-only unit test for the server's JSON input parser (issue #34) — no model/lib.
add_executable(test-server-json-input tests/test_server_json_input.cpp)
target_include_directories(test-server-json-input PRIVATE ${CMAKE_SOURCE_DIR}/examples/server
                                                          ${CMAKE_SOURCE_DIR}/src)

# Host-only unit test for arch-driven GGUF hparam key resolution (issue #33 / A2).
add_executable(test-hparam-keys tests/test_hparam_keys.cpp)
target_include_directories(test-hparam-keys PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(test-hmer tests/test_hmer.cpp)
target_link_libraries(test-hmer PRIVATE crispembed)

add_executable(test-bttr tests/test_bttr.cpp)
target_link_libraries(test-bttr PRIVATE crispembed)

add_executable(test-hmer-image tests/test_hmer_image.cpp)
target_link_libraries(test-hmer-image PRIVATE crispembed)

add_executable(test-bttr-beam tests/test_bttr_beam.cpp)
target_link_libraries(test-bttr-beam PRIVATE crispembed)

add_executable(test-posformer tests/test_posformer.cpp)
target_link_libraries(test-posformer PRIVATE crispembed)

add_executable(test-parseq tests/test_parseq.cpp)
target_link_libraries(test-parseq PRIVATE crispembed)

add_executable(test-easyocr tests/test_easyocr.cpp)
target_include_directories(test-easyocr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-easyocr PRIVATE crispembed)

add_executable(test-easyocr-diff tests/test_easyocr_diff.cpp)
target_include_directories(test-easyocr-diff PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-easyocr-diff PRIVATE crispembed)

add_executable(test-easyocr-benchmark tests/test_easyocr_benchmark.cpp)
target_include_directories(test-easyocr-benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-easyocr-benchmark PRIVATE crispembed)

add_executable(test-easyocr-dbnet tests/test_easyocr_dbnet.cpp)
target_include_directories(test-easyocr-dbnet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-easyocr-dbnet PRIVATE crispembed)

add_executable(test-dbnet-diff tests/test_dbnet_diff.cpp)
target_link_libraries(test-dbnet-diff PRIVATE crispembed)

add_executable(test-dbnet-benchmark tests/test_dbnet_benchmark.cpp)
target_link_libraries(test-dbnet-benchmark PRIVATE crispembed)

add_executable(test-easyocr-layout tests/test_easyocr_layout.cpp)
target_link_libraries(test-easyocr-layout PRIVATE crispembed)

add_executable(test-easyocr-postprocess tests/test_easyocr_postprocess.cpp)
target_link_libraries(test-easyocr-postprocess PRIVATE crispembed)

add_executable(test-easyocr-pipeline tests/test_easyocr_pipeline.cpp)
target_link_libraries(test-easyocr-pipeline PRIVATE crispembed)

add_executable(test-easyocr-craft-diff tests/test_easyocr_craft_diff.cpp)
target_include_directories(test-easyocr-craft-diff PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-easyocr-craft-diff PRIVATE crispembed)

add_executable(test-easyocr-craft-benchmark tests/test_easyocr_craft_benchmark.cpp)
target_include_directories(test-easyocr-craft-benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-easyocr-craft-benchmark PRIVATE crispembed)

if(NOT WIN32)
    add_executable(test-posformer-batch tests/test_posformer_batch.cpp)
    target_link_libraries(test-posformer-batch PRIVATE crispembed)
endif()

add_executable(test-ppformulanet tests/test_ppformulanet.cpp)
target_link_libraries(test-ppformulanet PRIVATE crispembed)

add_executable(test-ppformulanet-l tests/test_ppformulanet_l.cpp)
target_link_libraries(test-ppformulanet-l PRIVATE crispembed)

add_executable(test-mixtex tests/test_mixtex.cpp)
target_link_libraries(test-mixtex PRIVATE crispembed)

add_executable(test-mixtex-diff tests/test_mixtex_diff.cpp)
target_link_libraries(test-mixtex-diff PRIVATE crispembed)

add_executable(test-math-ocr-diff tests/test_math_ocr_diff.cpp)
target_link_libraries(test-math-ocr-diff PRIVATE crispembed)

add_executable(test-smt-diff tests/test_smt_diff.cpp)
target_link_libraries(test-smt-diff PRIVATE crispembed)

add_executable(test-transcoda-diff tests/test_transcoda_diff.cpp)
target_link_libraries(test-transcoda-diff PRIVATE crispembed)

add_executable(test-smt-reco tests/test_smt_reco.cpp)
target_link_libraries(test-smt-reco PRIVATE crispembed)

add_executable(test-tromr-diff tests/test_tromr_diff.cpp)
target_link_libraries(test-tromr-diff PRIVATE crispembed)

add_executable(test-flova-diff tests/test_flova_diff.cpp)
target_link_libraries(test-flova-diff PRIVATE crispembed)

add_executable(test-hat-diff tests/test_hat_diff.cpp)
target_link_libraries(test-hat-diff PRIVATE crispembed)

add_executable(test-surya-det tests/test_surya_det.cpp)
target_link_libraries(test-surya-det PRIVATE crispembed)

add_executable(test-surya-det-diff tests/test_surya_det_diff.cpp)
target_link_libraries(test-surya-det-diff PRIVATE crispembed)

add_executable(test-ocr-detect tests/test_ocr_detect.cpp)
target_link_libraries(test-ocr-detect PRIVATE crispembed)

add_executable(test-ocr-geometry tests/test_ocr_geometry.cpp)
target_link_libraries(test-ocr-geometry PRIVATE crispembed)

add_executable(test-ocr-crop tests/test_ocr_crop.cpp)
target_link_libraries(test-ocr-crop PRIVATE crispembed)

add_executable(test-ocr-detect-diff tests/test_ocr_detect_diff.cpp)
target_link_libraries(test-ocr-detect-diff PRIVATE crispembed)

add_executable(test-ppocrv6-det-diff tests/test_ppocrv6_det_diff.cpp)
target_link_libraries(test-ppocrv6-det-diff PRIVATE crispembed)

add_executable(test-ocr-region-router tests/test_ocr_region_router.cpp)
target_link_libraries(test-ocr-region-router PRIVATE crispembed)

add_executable(test-ocr-recognize tests/test_ocr_recognize.cpp)
target_link_libraries(test-ocr-recognize PRIVATE crispembed)

add_executable(test-layout-detect tests/test_layout_detect.cpp)
target_link_libraries(test-layout-detect PRIVATE crispembed)

if(NOT WIN32)
    add_executable(test-layout-diff tests/test_layout_diff.cpp)
    target_link_libraries(test-layout-diff PRIVATE crispembed)
endif()

add_executable(test-gliner-ner tests/test_gliner_ner.cpp)
target_link_libraries(test-gliner-ner PRIVATE crispembed)

add_executable(test-kie-pipeline tests/test_kie_pipeline.cpp)
target_link_libraries(test-kie-pipeline PRIVATE crispembed)

add_executable(test-bert-ner tests/test_bert_ner.cpp)
target_link_libraries(test-bert-ner PRIVATE crispembed)

add_executable(test-lightonocr tests/test_lightonocr.cpp)
target_link_libraries(test-lightonocr PRIVATE crispembed)

add_executable(test-lilt-kie tests/test_lilt_kie.cpp)
target_link_libraries(test-lilt-kie PRIVATE crispembed)

add_executable(test-lilt-diff tests/test_lilt_diff.cpp)
target_link_libraries(test-lilt-diff PRIVATE crispembed)

add_executable(test-gliner-diff tests/test_gliner_diff.cpp)
target_link_libraries(test-gliner-diff PRIVATE crispembed)

add_executable(test-scan-cleanup tests/test_scan_cleanup.cpp)
target_link_libraries(test-scan-cleanup PRIVATE crispembed)
add_executable(test-ocr-orchestrator tests/test_ocr_orchestrator.cpp)
target_link_libraries(test-ocr-orchestrator PRIVATE crispembed)

add_executable(test-conv-ext tests/test_conv_ext.cpp)
target_link_libraries(test-conv-ext PRIVATE crispembed)

add_executable(test-tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test-tokenizer PRIVATE crispembed)

# Hermetic: core/bpe.h is header-only, so this needs no model, vocab or ggml.
add_executable(test-qwen-pretokenize tests/test_qwen_pretokenize.cpp)
target_include_directories(test-qwen-pretokenize PRIVATE src)

# Hermetic: core/bpe.h + core/unicode_categ.h are header-only.
add_executable(test-o200k-pretokenize tests/test_o200k_pretokenize.cpp)
target_include_directories(test-o200k-pretokenize PRIVATE src)

add_executable(test-ministral-pretokenize tests/test_ministral_pretokenize.cpp)
target_include_directories(test-ministral-pretokenize PRIVATE src)

# Same, for the LFM2 and DeepSeek-OCR-2/Unlimited-OCR pre-tokenizers and the
# bpe_one merge tie-break. Regenerate with tools/gen_bpe_pretokenize_test.py.
add_executable(test-bpe-pretokenize tests/test_bpe_pretokenize.cpp)
target_include_directories(test-bpe-pretokenize PRIVATE src)

# Hermetic: core/bert_pretok.h HF-Bert pre-tokenization (LaBSE-class WordPiece,
# tokenizer.ggml.pre = "bert") + the tokenizer-family routing table.
add_executable(test-bert-pretokenize tests/test_bert_pretokenize.cpp)
target_include_directories(test-bert-pretokenize PRIVATE src)

# Hermetic: core/bert_norm.h — HF BertNormalizer's strip_accents+lowercase for
# uncased WordPiece models (the accented-Latin [UNK] fix). Goldens captured
# from HF's own Rust normalizer; also pins the ASCII invariant that makes the
# default safe. Links tokenizer.cpp for the end-to-end WordPiece cases.
add_executable(test-bert-norm tests/test_bert_norm.cpp src/tokenizer.cpp)
target_include_directories(test-bert-norm PRIVATE src)

# Hermetic: core/spm_norm.h — the SentencePiece Precompiled (nmt_nfkc
# charsmap) normalizer for XLM-R-family Unigram models. Goldens from HF's own
# Precompiled component; pins the printable-ASCII invariant and the \t/control
# handling that "ASCII is untouched" would misstate.
add_executable(test-spm-norm tests/test_spm_norm.cpp)
target_include_directories(test-spm-norm PRIVATE src)

# Live (needs a GGUF): decoded-output A/B for the fireredpunc tokenizer fix.
# Punctuates each input line; the arm is chosen by
# CRISPEMBED_FIREREDPUNC_HF_TOK, so one binary produces both.
add_executable(firered-punct-ab tests/firered_punct_ab.cpp)
target_include_directories(firered-punct-ab PRIVATE src)
target_link_libraries(firered-punct-ab PRIVATE crispembed)

# Live (needs a GGUF): dump the token ids the shipping runtime produces, for
# ANY tokenizer family. SentencePiece/BPE parity cannot be checked from a
# vocab file the way WordPiece can — merges, charsmap and pre-tokenizer
# selection live in the GGUF. Driven by tests/embed_tokenizer_parity.py.
add_executable(dump-token-ids tests/dump_token_ids.cpp)
target_include_directories(dump-token-ids PRIVATE src)
target_link_libraries(dump-token-ids PRIVATE crispembed)

# Hermetic: core/imatrix_alias.h — the F7 naming contract between the
# runtime's pre-merged QKV tensor and the quantizer's imatrix alias lookup.
add_executable(test-imatrix-alias tests/test_imatrix_alias.cpp)
target_include_directories(test-imatrix-alias PRIVATE src)

# Hermetic: core/no_repeat_ngram.h — the shared greedy no-repeat-ngram guard
# (deepseek-ocr2 contract ngram=20; qwen2vl/internvl2 ngram=3).
add_executable(test-no-repeat-ngram tests/test_no_repeat_ngram.cpp)
target_include_directories(test-no-repeat-ngram PRIVATE src)

# Hermetic: core/env_gate.h — the shared value-parsed boolean env gate that
# every CRISPEMBED_*_BENCH instrument routes through. Guards against the
# presence-based inversion where `FOO=0` ENABLED the gate.
add_executable(test-env-gate tests/test_env_gate.cpp)
target_include_directories(test-env-gate PRIVATE src)

add_executable(test-litemla-attn tests/test_litemla_attn.cpp)
target_link_libraries(test-litemla-attn PRIVATE crispembed)

add_executable(test-core-cpu-ops tests/test_core_cpu_ops.cpp)
target_link_libraries(test-core-cpu-ops PRIVATE crispembed)

add_executable(test-core-vlm-attention tests/test_core_vlm_attention.cpp)
target_link_libraries(test-core-vlm-attention PRIVATE crispembed)

add_executable(test-lfm2-shortconv tests/test_lfm2_shortconv.cpp)
target_link_libraries(test-lfm2-shortconv PRIVATE crispembed)

add_executable(test-msac-tiling tests/test_msac_tiling.cpp)
target_link_libraries(test-msac-tiling PRIVATE crispembed)

add_executable(test-provenance-marking tests/test_provenance_marking.cpp)
target_include_directories(test-provenance-marking PRIVATE src)

add_executable(test-temp-file tests/test_temp_file.cpp)
target_include_directories(test-temp-file PRIVATE src)

add_executable(test-render-provenance tests/test_render_provenance.cpp)
target_link_libraries(test-render-provenance PRIVATE crispembed)
target_compile_definitions(test-render-provenance PRIVATE CRISPEMBED_VERSION_STR="${CRISPEMBED_VERSION_RAW}")

add_executable(test-image-provenance tests/test_image_provenance.cpp)
target_link_libraries(test-image-provenance PRIVATE crispembed-core)
crispembed_enable_c2pa(test-image-provenance)

add_executable(test-gguf-loader-mmap tests/test_gguf_loader_mmap.cpp)
target_link_libraries(test-gguf-loader-mmap PRIVATE crispembed)

add_executable(test-ocr-pipeline tests/test_ocr_pipeline.cpp)
target_link_libraries(test-ocr-pipeline PRIVATE crispembed)

add_executable(test-ocr-pipeline-pool tests/test_ocr_pipeline_pool.cpp)
target_link_libraries(test-ocr-pipeline-pool PRIVATE crispembed)

add_executable(test-qwen2vl tests/test_qwen2vl.cpp)
target_link_libraries(test-qwen2vl PRIVATE crispembed)

add_executable(test-qwen2vl-diff tests/test_qwen2vl_diff.cpp)
target_link_libraries(test-qwen2vl-diff PRIVATE crispembed)

add_executable(test-paddleocr-vl-diff tests/test_paddleocr_vl_diff.cpp)
target_link_libraries(test-paddleocr-vl-diff PRIVATE crispembed)

add_executable(test-qwen2vl-e2e tests/test_qwen2vl_e2e.cpp)
target_link_libraries(test-qwen2vl-e2e PRIVATE crispembed)

add_executable(test-internvl2-diff tests/test_internvl2_diff.cpp)
target_link_libraries(test-internvl2-diff PRIVATE crispembed)

add_executable(test-internvl2-e2e tests/test_internvl2_e2e.cpp)
target_link_libraries(test-internvl2-e2e PRIVATE crispembed)

add_executable(test-internvl2-image tests/test_internvl2_image.cpp)
target_link_libraries(test-internvl2-image PRIVATE crispembed)

add_executable(test-glm-ocr-diff tests/test_glm_ocr_diff.cpp)
target_link_libraries(test-glm-ocr-diff PRIVATE crispembed)

add_executable(test-glm-ocr-e2e tests/test_glm_ocr_e2e.cpp)
target_link_libraries(test-glm-ocr-e2e PRIVATE crispembed)

add_executable(test-glm-ocr-image tests/test_glm_ocr_image.cpp)
target_link_libraries(test-glm-ocr-image PRIVATE crispembed)

add_executable(test-got-ocr-diff tests/test_got_ocr_diff.cpp)
target_link_libraries(test-got-ocr-diff PRIVATE crispembed)

# Live (needs a GGUF): dump clip_text/SigLIP token ids for the HF parity
# harness. clip_text is a standalone engine, so the generic dump-token-ids
# driver cannot load its GGUFs.
add_executable(clip-text-dump-ids tests/clip_text_dump_ids.cpp)
target_include_directories(clip-text-dump-ids PRIVATE src)
target_link_libraries(clip-text-dump-ids PRIVATE crispembed)

add_executable(test-clip-text-diff tests/test_clip_text_diff.cpp)
target_link_libraries(test-clip-text-diff PRIVATE crispembed)

add_executable(test-tesseract-lstm-diff tests/test_tesseract_lstm_diff.cpp)
target_include_directories(test-tesseract-lstm-diff PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-tesseract-lstm-diff PRIVATE crispembed)

add_executable(test-ocr-identical-crops tests/test_ocr_identical_crops.cpp)
target_include_directories(test-ocr-identical-crops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-ocr-identical-crops PRIVATE crispembed)

add_executable(test-morph-fast tests/test_morph_fast.cpp)
target_link_libraries(test-morph-fast PRIVATE crispembed)

add_executable(test-cc-detect tests/test_cc_detect.cpp)
target_include_directories(test-cc-detect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-cc-detect PRIVATE crispembed)

add_executable(test-classical-preproc tests/test_classical_preproc.cpp)
target_include_directories(test-classical-preproc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-classical-preproc PRIVATE crispembed)

add_executable(test-ocr-render tests/test_ocr_render.cpp)
target_link_libraries(test-ocr-render PRIVATE crispembed)

add_executable(test-dewarp tests/test_dewarp.cpp)
target_include_directories(test-dewarp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-dewarp PRIVATE crispembed)

add_executable(test-tps-warp tests/test_tps_warp.cpp)
target_link_libraries(test-tps-warp PRIVATE crispembed)

add_executable(test-bert-ner-diff tests/test_bert_ner_diff.cpp)
target_link_libraries(test-bert-ner-diff PRIVATE crispembed)

add_executable(test-modernbert-diff tests/test_modernbert_diff.cpp)
target_link_libraries(test-modernbert-diff PRIVATE crispembed)

add_executable(test-tps-locnet tests/test_tps_locnet.cpp)
target_link_libraries(test-tps-locnet PRIVATE crispembed)

add_executable(test-tps-parity tests/test_tps_parity.cpp)
target_link_libraries(test-tps-parity PRIVATE crispembed)

add_executable(test-text-sr tests/test_text_sr.cpp)
target_link_libraries(test-text-sr PRIVATE crispembed)

add_executable(test-tbsrn-diff tests/test_tbsrn_diff.cpp)
target_link_libraries(test-tbsrn-diff PRIVATE crispembed)

add_executable(test-safmn-diff tests/test_safmn_diff.cpp)
target_link_libraries(test-safmn-diff PRIVATE crispembed)

add_executable(test-esrgan-diff tests/test_esrgan_diff.cpp)
target_link_libraries(test-esrgan-diff PRIVATE crispembed)

add_executable(test-instructir-diff tests/test_instructir_diff.cpp)
target_link_libraries(test-instructir-diff PRIVATE crispembed)

add_executable(test-scunet-diff tests/test_scunet_diff.cpp)
target_link_libraries(test-scunet-diff PRIVATE crispembed)

add_executable(test-pan-diff tests/test_pan_diff.cpp)
target_link_libraries(test-pan-diff PRIVATE crispembed)

add_executable(test-restormer-diff tests/test_restormer_diff.cpp)
target_link_libraries(test-restormer-diff PRIVATE crispembed)

add_executable(test-nafnet-diff tests/test_nafnet_diff.cpp)
target_link_libraries(test-nafnet-diff PRIVATE crispembed)

add_executable(test-vit-embed-diff tests/test_vit_embed_diff.cpp)
target_link_libraries(test-vit-embed-diff PRIVATE crispembed)

add_executable(test-bidirlm-vision-diff tests/test_bidirlm_vision_diff.cpp)
target_link_libraries(test-bidirlm-vision-diff PRIVATE crispembed)

add_executable(test-punct-diff tests/test_punct_diff.cpp)
target_link_libraries(test-punct-diff PRIVATE crispembed)

add_executable(test-face-diff tests/test_face_diff.cpp)
target_link_libraries(test-face-diff PRIVATE crispembed)

add_executable(test-decoder-embed-diff tests/test_decoder_embed_diff.cpp)
target_link_libraries(test-decoder-embed-diff PRIVATE crispembed)

add_executable(test-pdf-info tests/test_pdf_info.cpp)
target_link_libraries(test-pdf-info PRIVATE crispembed)

add_executable(test-table-parse tests/test_table_parse.cpp)
target_link_libraries(test-table-parse PRIVATE crispembed)

add_executable(test-swinir-diff tests/test_swinir_diff.cpp)
target_link_libraries(test-swinir-diff PRIVATE crispembed)

add_executable(test-dat-diff tests/test_dat_diff.cpp)
target_link_libraries(test-dat-diff PRIVATE crispembed)

add_executable(test-granite-vision-diff tests/test_granite_vision_diff.cpp)
target_link_libraries(test-granite-vision-diff PRIVATE crispembed)

add_executable(test-smoldocling-smoke tests/test_smoldocling_smoke.cpp)
target_link_libraries(test-smoldocling-smoke PRIVATE crispembed)

add_executable(test-smoldocling-diff tests/test_smoldocling_diff.cpp)
target_link_libraries(test-smoldocling-diff PRIVATE crispembed)

add_executable(test-confidence tests/test_confidence.cpp)
target_link_libraries(test-confidence PRIVATE crispembed)

add_executable(test-tesseract-dawg tests/test_tesseract_dawg.cpp)
target_link_libraries(test-tesseract-dawg PRIVATE crispembed)

add_executable(test-tesseract-dawg-runtime tests/test_tesseract_dawg_runtime.cpp)
target_include_directories(test-tesseract-dawg-runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test-tesseract-dawg-runtime PRIVATE crispembed)

add_executable(test-tesseract-recorder tests/test_tesseract_recoder.cpp)
target_include_directories(test-tesseract-recorder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test-tesseract-recorder PRIVATE crispembed)

add_executable(test-tesseract-dawg-score tests/test_tesseract_dawg_score.cpp)
target_include_directories(test-tesseract-dawg-score PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test-tesseract-dawg-score PRIVATE crispembed)

add_executable(test-ppocrv6-rec tests/test_ppocrv6_rec.cpp)
target_link_libraries(test-ppocrv6-rec PRIVATE crispembed)

add_executable(test-ppocrv6-dbnet tests/test_ppocrv6_dbnet.cpp)
target_include_directories(test-ppocrv6-dbnet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-ppocrv6-dbnet PRIVATE crispembed)

add_executable(test-ppocrv6-direct tests/test_ppocrv6_direct.cpp)
target_include_directories(test-ppocrv6-direct PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-ppocrv6-direct PRIVATE crispembed)

add_executable(test-pplcnet-orientation tests/test_pplcnet_orientation.cpp)
target_include_directories(test-pplcnet-orientation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
target_link_libraries(test-pplcnet-orientation PRIVATE crispembed)

add_executable(test-backend-smoke tests/test_backend_smoke.cpp)
target_link_libraries(test-backend-smoke PRIVATE crispembed)

endif() # NOT iOS

# ---------------------------------------------------------------------------
# WASM target — math OCR for browser use (Emscripten only).
# Built via: ./build-wasm.sh (which sets CRISPEMBED_WASM=ON)
# Output: crispembed_ocr.js + crispembed_ocr.wasm
# ---------------------------------------------------------------------------
option(CRISPEMBED_WASM "Build WASM math OCR target (Emscripten)" OFF)
if(CRISPEMBED_WASM AND EMSCRIPTEN)
    add_executable(crispembed-wasm wasm/ocr_wrapper.c)
    target_include_directories(crispembed-wasm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
    target_link_libraries(crispembed-wasm PRIVATE crispembed)
    set_target_properties(crispembed-wasm PROPERTIES
        OUTPUT_NAME  crispembed_ocr
        SUFFIX       ".js"
    )
endif()

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
    crispembed_apply_arch_flags(crispembed-shared)
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
    if(CRISPEMBED_HAS_CRISP_PUNC)
        target_link_libraries(crispembed-shared PRIVATE crisp_punc)
    endif()
    if(CRISPEMBED_HAS_CRISP_LID)
        target_link_libraries(crispembed-shared PRIVATE crisp_lid)
    endif()
    if(CRISPEMBED_HAS_CRISP_TRUECASE)
        target_link_libraries(crispembed-shared PRIVATE crisp_truecase)
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

option(CRISPEMBED_WASM_EMBED "Build WASM text-embedding target (Emscripten)" OFF)
if(CRISPEMBED_WASM_EMBED AND EMSCRIPTEN)
    add_executable(crispembed-embed-wasm wasm/embed_wrapper.c)
    target_include_directories(crispembed-embed-wasm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
    target_link_libraries(crispembed-embed-wasm PRIVATE crispembed)
    set_target_properties(crispembed-embed-wasm PROPERTIES
        OUTPUT_NAME  crispembed_embed
        SUFFIX       ".js"
    )
endif()
add_executable(test-adair-diff tests/test_adair_diff.cpp)
target_link_libraries(test-adair-diff PRIVATE crispembed)

add_executable(test-pix2struct-diff tests/test_pix2struct_diff.cpp)
target_link_libraries(test-pix2struct-diff PRIVATE crispembed)

add_executable(test-lfm2-diff tests/test_lfm2_diff.cpp)
target_link_libraries(test-lfm2-diff PRIVATE crispembed)

add_executable(test-lfm2-colbert-diff tests/test_lfm2_colbert_diff.cpp)
target_link_libraries(test-lfm2-colbert-diff PRIVATE crispembed)
