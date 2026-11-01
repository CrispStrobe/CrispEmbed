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
    # Adopting crisp_punc REPLACES src/fireredpunc.cpp + src/pcs.cpp. The two
    # fireredpunc copies HAD diverged, so this silently swapped the punctuation
    # restorer for one with a known tokenizer defect (issue #50). They have since
    # been reconciled in both directions — CrispEmbed's HF WordPiece unification
    # went upstream, CrispASR's #300 no-stacking guard came downstream into the
    # shared core/punct_marks.h — and the two now produce byte-identical output
    # and byte-identical token ids on the same corpora. If you change one copy,
    # change the other; the sibling-crispasr CI job builds this configuration.
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
    src/core/shortest_double.c
)
crispembed_enable_c2pa(crispembed-core)
target_include_directories(crispembed-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_link_libraries(crispembed-core PUBLIC ggml ggml-base)
target_include_directories(crispembed-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core/third_party)

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
list(APPEND CRISPEMBED_SOURCES src/mineru_ocr.cpp src/mineru_document.cpp src/mineru_postprocess.cpp
     src/mineru_table_image.cpp)
# Exact Python float semantics (bbox ratios, luminance): never fuse mul+add.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/mineru_document.cpp src/mineru_postprocess.cpp src/mineru_table_image.cpp
                                PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()
list(APPEND CRISPEMBED_SOURCES src/oxr_ocr.cpp src/oxr_api.cpp src/oxr_layout_api.cpp)
list(APPEND CRISPEMBED_SOURCES src/internvl2_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/gliner_ner.cpp)
list(APPEND CRISPEMBED_SOURCES src/lfm2_embed.cpp src/gemma_decision.cpp src/ultrabert_embed.cpp)
list(APPEND CRISPEMBED_SOURCES src/josev2b_backbone.cpp src/josev2b_head.cpp
    src/josev2b_text_frontend.cpp src/josev2b_answer_serializer.cpp
    src/josev2b_composed.cpp src/systemone_api.cpp)
# Match the audited scalar operation order in the composed Josev CPU profile.
# Qwen helpers are shared with OCR; final OCR qualification must use this source.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_property(SOURCE src/josev2b_backbone.cpp src/josev2b_head.cpp
        src/josev2b_text_frontend.cpp src/josev2b_answer_serializer.cpp
        src/josev2b_composed.cpp src/qwen35_decoder.cpp src/qwen35_attention.cpp
        src/qwen35_linear.cpp src/qwen35_delta.cpp src/qwen35_norm.cpp
        src/qwen35_positions.cpp src/qwen35_profile.cpp
        APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)
endif()
list(APPEND CRISPEMBED_SOURCES src/scan_cleanup.cpp)
list(APPEND CRISPEMBED_SOURCES src/nafnet_denoise.cpp)
list(APPEND CRISPEMBED_SOURCES src/text_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/tbsrn_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/pan_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/sam2.cpp src/sam2_capi.cpp)
list(APPEND CRISPEMBED_SOURCES src/hat_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/dat_sr.cpp)
list(APPEND CRISPEMBED_SOURCES src/deepseek_ocr2.cpp)
list(APPEND CRISPEMBED_SOURCES src/unlimited_ocr.cpp)
list(APPEND CRISPEMBED_SOURCES src/unlimited_ocr2_dash.cpp)
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
list(APPEND CRISPEMBED_SOURCES src/qwen35_delta.cpp src/qwen35_linear.cpp src/qwen35_attention.cpp src/qwen35_decoder.cpp src/qwen35_vision.cpp src/qwen35_norm.cpp src/qwen35_profile.cpp src/qwen35_positions.cpp src/qwen35_image.cpp src/qwen35_tokenizer.cpp src/qwen35_ocr.cpp)
# Experimental MiniCPM-OCR-1.0 (default-off: CRISPEMBED_MINICPM_OCR); reuses the qwen35 decoder.
list(APPEND CRISPEMBED_SOURCES src/minicpm_ocr_vision.cpp src/minicpm_ocr.cpp)
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
list(APPEND CRISPEMBED_SOURCES src/marker.cpp src/marker_document.cpp src/marker_backend.cpp)
# The shared target compiles these sources separately from the static library.
# Keep Marker provenance identical in both, without changing other backends.
set_source_files_properties(src/marker.cpp PROPERTIES
    COMPILE_DEFINITIONS CRISPEMBED_VERSION_STR="${CRISPEMBED_VERSION_RAW}")
list(APPEND CRISPEMBED_SOURCES src/table_parse.cpp)
# MoEdern1.5M raw-text/role public API: default-off gates, exact arithmetic.
set(CRISPEMBED_MOEDERN_SOURCES src/moedern_text_engine.cpp src/moedern_text_frontend.cpp src/moedern_family_engine.cpp
    src/moedern_text.cpp src/moedern_text_api.cpp)
list(APPEND CRISPEMBED_SOURCES ${CRISPEMBED_MOEDERN_SOURCES})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(${CRISPEMBED_MOEDERN_SOURCES} PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()
# 27M/115M dispatch (runtime gate CRISPEMBED_MOEDERN_FAMILY, default off).
set_source_files_properties(src/moedern_text.cpp PROPERTIES COMPILE_DEFINITIONS CRISPEMBED_MOEDERN_FAMILY_ENGINE)

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
    ${CMAKE_DL_LIBS}
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
    examples/cli/inpaint_cli.cpp
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

add_executable(crispembed-marker examples/marker/main.cpp)
target_link_libraries(crispembed-marker PRIVATE crispembed)
set_target_properties(crispembed-marker PROPERTIES INSTALL_RPATH "${CRISPEMBED_BIN_INSTALL_RPATH}")
install(TARGETS crispembed-marker RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

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

# Host-only: the "never stack punctuation on punctuation" predicate (#300).
add_executable(test-punct-marks tests/test_punct_marks.cpp)
target_include_directories(test-punct-marks PRIVATE src)

# Host-only: the files CrispEmbed mirrors from CrispASR must not drift (#50).
# Skips unless CRISPASR_SIBLING_DIR points at a real checkout, so a standalone
# build is unaffected; the sibling-crispasr CI job supplies it.
add_executable(test-cross-repo-sync tests/test_cross_repo_sync.cpp)
target_include_directories(test-cross-repo-sync PRIVATE src)

# Host-only unit test for arch-driven GGUF hparam key resolution (issue #33 / A2).
add_executable(test-hparam-keys tests/test_hparam_keys.cpp)
target_include_directories(test-hparam-keys PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(test-unlimited-ocr2-tiles tests/test_unlimited_ocr2_tiles.cpp)
target_include_directories(test-unlimited-ocr2-tiles PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_executable(test-minicpm-ocr-tiles tests/test_minicpm_ocr_tiles.cpp)
target_include_directories(test-minicpm-ocr-tiles PRIVATE ${CMAKE_SOURCE_DIR}/src)

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
#
# Builds in BOTH layouts now. It used to fail to link against a sibling
# CrispASR with `undefined reference to fireredpunc_debug_token_ids`, because
# adopting crisp_punc drops src/fireredpunc.cpp and the shared copy had neither
# that hook nor the tokenizer gate this A/Bs. Both were ported upstream, so the
# two copies expose the same surface and this target is layout-independent.
add_executable(firered-punct-ab tests/firered_punct_ab.cpp)
target_include_directories(firered-punct-ab PRIVATE src)
target_link_libraries(firered-punct-ab PRIVATE crispembed)

# Live (needs a GGUF): drives ANY punctuation model through the public C API,
# which dispatches on general.architecture — so it reaches PCS, which
# firered-punct-ab cannot (it calls fireredpunc_* directly).
add_executable(punct-ab tests/punct_ab.cpp)
target_include_directories(punct-ab PRIVATE src)
target_link_libraries(punct-ab PRIVATE crispembed)

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

# Experimental Qwen3.5 recurrence boundary, independently testable before the
# full OCR engine is wired. No production C ABI or registry exposure yet.
add_library(crispembed-qwen35-delta STATIC src/qwen35_delta.cpp src/qwen35_linear.cpp src/qwen35_attention.cpp src/qwen35_decoder.cpp src/qwen35_vision.cpp src/qwen35_norm.cpp src/qwen35_profile.cpp src/qwen35_positions.cpp src/qwen35_image.cpp src/qwen35_tokenizer.cpp src/qwen35_ocr.cpp)
target_include_directories(crispembed-qwen35-delta PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# qwen35_linear.cpp uses std::thread; older manylinux glibc needs explicit pthread linkage.
target_link_libraries(crispembed-qwen35-delta PUBLIC ggml ggml-base ggml-cpu crispembed-core Threads::Threads)
add_executable(test-qwen35-public tests/test_qwen35_public.cpp)
target_link_libraries(test-qwen35-public PRIVATE crispembed crispembed-core)
add_executable(test-qwen35-context tests/test_qwen35_context.cpp tests/qwen35_context_header.c)
target_link_libraries(test-qwen35-context PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-qwen35-quant tests/test_qwen35_quant.cpp)
target_link_libraries(test-qwen35-quant PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-qwen35-materialize tests/test_qwen35_materialize.cpp)
target_link_libraries(test-qwen35-materialize PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-qwen35-weight-cache tests/test_qwen35_weight_cache.cpp)
target_link_libraries(test-qwen35-weight-cache PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-ovisocr2-cache-bench tests/test_ovisocr2_cache_bench.cpp)
target_link_libraries(test-ovisocr2-cache-bench PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-qwen35-profile tests/test_qwen35_profile.cpp)
target_link_libraries(test-qwen35-profile PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-qwen35-imatrix tests/test_qwen35_imatrix.cpp)
target_link_libraries(test-qwen35-imatrix PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-qwen35-delta tests/test_qwen35_delta.cpp)
target_link_libraries(test-qwen35-delta PRIVATE crispembed-qwen35-delta)
add_executable(test-qwen35-linear tests/test_qwen35_linear.cpp)
target_link_libraries(test-qwen35-linear PRIVATE crispembed-qwen35-delta crispembed-core)

add_executable(test-lfm2-tiling tests/test_lfm2_tiling.cpp)
target_link_libraries(test-lfm2-tiling PRIVATE crispembed)

add_executable(test-pil-resize tests/test_pil_resize.cpp)
target_link_libraries(test-pil-resize PRIVATE crispembed)

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

add_executable(test-mineru-doc tests/test_mineru_doc.cpp)
target_link_libraries(test-mineru-doc PRIVATE crispembed)
target_include_directories(test-mineru-doc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(test-otsl-table tests/test_otsl_table.cpp)
target_link_libraries(test-otsl-table PRIVATE crispembed)
target_include_directories(test-otsl-table PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(test-mineru-layout tests/test_mineru_layout.cpp)
target_include_directories(test-mineru-layout PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(test-mineru25-run tests/mineru25_run.cpp)
target_link_libraries(test-mineru25-run PRIVATE crispembed)
target_include_directories(test-mineru25-run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
add_executable(test-mineru25-diff tests/mineru25_diff.cpp)
target_link_libraries(test-mineru25-diff PRIVATE crispembed)
add_executable(test-mineru25-preprocess tests/mineru25_preprocess.cpp)
target_link_libraries(test-mineru25-preprocess PRIVATE crispembed)
target_include_directories(test-mineru25-preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
add_executable(test-mineru25-decode-graph tests/test_mineru25_decode_graph.cpp)
target_link_libraries(test-mineru25-decode-graph PRIVATE crispembed)
add_executable(test-mineru25-profile tests/test_mineru25_profile.cpp)
target_include_directories(test-mineru25-profile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(test-oxr-profile tests/test_oxr_profile.cpp)
target_include_directories(test-oxr-profile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(test-oxr-graph tests/test_oxr_graph.cpp)
target_link_libraries(test-oxr-graph PRIVATE crispembed)
add_executable(test-oxr-api tests/test_oxr_api.cpp src/oxr_api.cpp)
target_include_directories(test-oxr-api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(test-oxr-run tests/oxr_run.cpp)
target_link_libraries(test-oxr-run PRIVATE crispembed)
target_include_directories(test-oxr-run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)
add_executable(test-oxr-dfine-unit tests/test_oxr_dfine_unit.cpp)
target_link_libraries(test-oxr-dfine-unit PRIVATE crispembed)
add_executable(test-oxr-dfine-diff tests/oxr_dfine_diff.cpp)
target_link_libraries(test-oxr-dfine-diff PRIVATE crispembed)
add_executable(test-oxr-diff tests/oxr_diff.cpp)
target_link_libraries(test-oxr-diff PRIVATE crispembed)
target_include_directories(test-oxr-diff PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ggml/examples)

add_executable(test-qwen2vl-diff tests/test_qwen2vl_diff.cpp)
target_link_libraries(test-qwen2vl-diff PRIVATE crispembed)

# TeleOCR observers exist only in this validation translation unit. The normal
# static/shared libraries contain no observer setter or callback implementation.
add_executable(test-teleocr-diff tests/test_teleocr_diff.cpp src/qwen2vl_ocr.cpp)
target_compile_definitions(test-teleocr-diff PRIVATE CRISPEMBED_TELEOCR_TESTING=1)
crispembed_apply_arch_flags(test-teleocr-diff)
target_link_libraries(test-teleocr-diff PRIVATE crispembed)

add_executable(test-teleocr-preprocess tests/test_teleocr_preprocess.cpp)
target_link_libraries(test-teleocr-preprocess PRIVATE crispembed)

add_executable(test-marker-document tests/test-marker-document.cpp)
target_link_libraries(test-marker-document PRIVATE crispembed)

add_executable(test-marker-api tests/test-marker-api.cpp)
target_link_libraries(test-marker-api PRIVATE crispembed)

add_executable(test-marker-backend tests/test-marker-backend.cpp)
target_link_libraries(test-marker-backend PRIVATE crispembed)

add_executable(test-marker-save tests/test-marker-save.cpp)
target_include_directories(test-marker-save PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(test-teleocr-ffn tests/test_teleocr_ffn.cpp)
target_link_libraries(test-teleocr-ffn PRIVATE crispembed)

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

# SAM 2.1 (image mode). crispembed-sam2 is the engine alone: what applications that only segment
# link (static, or shared with CRISPEMBED_SAM2_SHARED), and what its parity test needs.
add_library(crispembed-sam2 STATIC src/sam2.cpp src/sam2_capi.cpp)
target_include_directories(crispembed-sam2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include)
crispembed_apply_arch_flags(crispembed-sam2)
target_link_libraries(crispembed-sam2 PUBLIC crispembed-core ggml ggml-base ggml-cpu)
option(CRISPEMBED_SAM2_SHARED "Also build libcrispembed-sam2 as a shared library" OFF)
if(CRISPEMBED_SAM2_SHARED)
    add_library(crispembed-sam2-shared SHARED src/sam2.cpp src/sam2_capi.cpp)
    set_target_properties(crispembed-sam2-shared PROPERTIES OUTPUT_NAME crispembed-sam2 POSITION_INDEPENDENT_CODE ON)
    target_include_directories(crispembed-sam2-shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include)
    target_compile_definitions(crispembed-sam2-shared PRIVATE CRISPEMBED_BUILD)
    crispembed_apply_arch_flags(crispembed-sam2-shared)
    target_link_libraries(crispembed-sam2-shared PRIVATE crispembed-core ggml ggml-base ggml-cpu)
endif()
add_executable(test-sam2-diff tests/test_sam2_diff.cpp)
target_link_libraries(test-sam2-diff PRIVATE crispembed-sam2)

# Fixed-256/512 MI-GAN CPU/F32 graph. The opt-in CLI links the ordinary archive;
# reference observers remain confined to the separate validation archive.
# Shared inpaint-v1 precision-variant loader (gated; F32 files bypass it).
add_library(crispembed-inpaint-weights STATIC EXCLUDE_FROM_ALL src/inpaint_weights.cpp)
target_include_directories(crispembed-inpaint-weights PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
crispembed_apply_arch_flags(crispembed-inpaint-weights)
target_link_libraries(crispembed-inpaint-weights PUBLIC crispembed-core ggml ggml-base ggml-cpu)
add_library(crispembed-migan STATIC EXCLUDE_FROM_ALL src/migan.cpp)
target_include_directories(crispembed-migan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
crispembed_apply_arch_flags(crispembed-migan)
target_link_libraries(crispembed-migan PUBLIC crispembed-inpaint-weights crispembed-core ggml ggml-base ggml-cpu)

# Issue 71: native-resolution tiling around the unchanged MI-GAN graph.
# This internal CLI library adds no shared-library ABI or model registry alias.
add_library(crispembed-inpaint STATIC EXCLUDE_FROM_ALL src/inpaint.cpp)
target_include_directories(crispembed-inpaint PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
crispembed_apply_arch_flags(crispembed-inpaint)
target_link_libraries(crispembed-inpaint PUBLIC crispembed-migan)
target_link_libraries(crispembed-cli PRIVATE crispembed-inpaint)

# Model-free contracts use the real tile/composition/hash implementation.
add_executable(test-inpaint EXCLUDE_FROM_ALL tests/test_inpaint.cpp)
target_link_libraries(test-inpaint PRIVATE crispembed-inpaint)

# Exercise the actual CLI parser/image/batch/output code with test-only model
# and detector implementations. This driver is never installed or distributed.
add_executable(test-inpaint-cli-driver EXCLUDE_FROM_ALL
    tests/inpaint_cli_driver.cpp examples/cli/inpaint_cli.cpp)
target_include_directories(test-inpaint-cli-driver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli)
target_link_libraries(test-inpaint-cli-driver PRIVATE crispembed-core)

# Observer code is compiled solely into this separate validation archive.
add_library(crispembed-migan-test STATIC EXCLUDE_FROM_ALL src/migan.cpp)
target_include_directories(crispembed-migan-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(crispembed-migan-test PRIVATE CRISPEMBED_MIGAN_TESTING)
crispembed_apply_arch_flags(crispembed-migan-test)
target_link_libraries(crispembed-migan-test PUBLIC crispembed-inpaint-weights crispembed-core ggml ggml-base ggml-cpu)
add_executable(test-migan-diff EXCLUDE_FROM_ALL tests/test_migan_diff.cpp)
target_link_libraries(test-migan-diff PRIVATE crispembed-migan-test)
add_executable(test-migan512-diff EXCLUDE_FROM_ALL tests/test_migan512_diff.cpp)
target_link_libraries(test-migan512-diff PRIVATE crispembed-migan-test)

# Issue71 latent inpainters share the existing CPU/GGUF/image infrastructure.
# Their observers are compiled only into the separate acceptance archive.
set(CRISPEMBED_LATENT_INPAINT_SOURCES
    src/latent_inpaint.cpp src/latent_inpaint_ops.cpp
    src/latent_inpaint_vae.cpp src/latent_inpaint_unet.cpp)
add_library(crispembed-latent-inpaint STATIC EXCLUDE_FROM_ALL ${CRISPEMBED_LATENT_INPAINT_SOURCES})
target_include_directories(crispembed-latent-inpaint PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
crispembed_apply_arch_flags(crispembed-latent-inpaint)
target_link_libraries(crispembed-latent-inpaint PUBLIC crispembed-inpaint crispembed-core ggml ggml-base ggml-cpu)
target_link_libraries(crispembed-cli PRIVATE crispembed-latent-inpaint)

add_library(crispembed-latent-inpaint-test STATIC EXCLUDE_FROM_ALL ${CRISPEMBED_LATENT_INPAINT_SOURCES})
target_include_directories(crispembed-latent-inpaint-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(crispembed-latent-inpaint-test PUBLIC CRISPEMBED_LATENT_INPAINT_TESTING)
crispembed_apply_arch_flags(crispembed-latent-inpaint-test)
target_link_libraries(crispembed-latent-inpaint-test PUBLIC crispembed-inpaint crispembed-core ggml ggml-base ggml-cpu)
add_executable(test-latent-inpaint EXCLUDE_FROM_ALL tests/test_latent_inpaint.cpp)
target_link_libraries(test-latent-inpaint PRIVATE crispembed-latent-inpaint-test)
add_executable(test-latent-inpaint-attention EXCLUDE_FROM_ALL tests/test_latent_inpaint_attention.cpp)
target_link_libraries(test-latent-inpaint-attention PRIVATE crispembed-latent-inpaint)
add_executable(test-latent-inpaint-pipeline EXCLUDE_FROM_ALL tests/test_latent_inpaint_pipeline.cpp)
target_link_libraries(test-latent-inpaint-pipeline PRIVATE crispembed-latent-inpaint)
add_executable(test-latent-inpaint-ops EXCLUDE_FROM_ALL tests/test_latent_inpaint_ops.cpp)
target_link_libraries(test-latent-inpaint-ops PRIVATE crispembed-latent-inpaint)

# Big-LaMa reuses the same bounded convolution/BN engine and adds its exact
# CPU Fourier path. Ordinary builds contain no reference observer callbacks.
add_library(crispembed-lama STATIC EXCLUDE_FROM_ALL src/lama.cpp src/lama_fft.cpp)
target_include_directories(crispembed-lama PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
crispembed_apply_arch_flags(crispembed-lama)
target_link_libraries(crispembed-lama PUBLIC crispembed-latent-inpaint)
target_link_libraries(crispembed-cli PRIVATE crispembed-lama)
add_library(crispembed-lama-test STATIC EXCLUDE_FROM_ALL src/lama.cpp src/lama_fft.cpp)
target_include_directories(crispembed-lama-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(crispembed-lama-test PUBLIC CRISPEMBED_LAMA_TESTING)
crispembed_apply_arch_flags(crispembed-lama-test)
target_link_libraries(crispembed-lama-test PUBLIC crispembed-latent-inpaint)
add_executable(test-lama-ops EXCLUDE_FROM_ALL tests/test_lama_ops.cpp)
target_link_libraries(test-lama-ops PRIVATE crispembed-lama-test)
add_executable(test-lama-diff EXCLUDE_FROM_ALL tests/test_lama_diff.cpp)
target_link_libraries(test-lama-diff PRIVATE crispembed-lama-test)

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
        crispembed-core ggml ggml-base ggml-cpu Threads::Threads ${CMAKE_DL_LIBS}
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
        PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/crispembed.h;${CMAKE_CURRENT_SOURCE_DIR}/src/moedern_text_api.h"
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

# Static CLI installations also redistribute the Apache-derived integrations.
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE
    ${CMAKE_CURRENT_SOURCE_DIR}/NOTICE.teleocr
    ${CMAKE_CURRENT_SOURCE_DIR}/NOTICE.marker
    ${CMAKE_CURRENT_SOURCE_DIR}/licenses/oxr-LICENSE
    ${CMAKE_CURRENT_SOURCE_DIR}/licenses/oxr-NOTICE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/crispembed
    COMPONENT licenses
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/models/licenses/apache-2.0/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/crispembed
    RENAME Apache-2.0.txt
    COMPONENT licenses
)

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
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
# Standalone validation executables follow the same platform rule as the CLI
# and tests above. The Qwen probe library is defined inside that native block.
add_executable(test-adair-diff tests/test_adair_diff.cpp)
target_link_libraries(test-adair-diff PRIVATE crispembed)

add_executable(test-pix2struct-diff tests/test_pix2struct_diff.cpp)
target_link_libraries(test-pix2struct-diff PRIVATE crispembed)

add_executable(test-lfm2-diff tests/test_lfm2_diff.cpp)
target_link_libraries(test-lfm2-diff PRIVATE crispembed)

add_executable(test-lfm2-colbert-diff tests/test_lfm2_colbert_diff.cpp)
target_link_libraries(test-lfm2-colbert-diff PRIVATE crispembed)

# Weight-free regression for FP32 activation arithmetic with quantized weights.
add_executable(test-quant-f32-matmul tests/test_quant_f32_matmul.cpp)
target_include_directories(test-quant-f32-matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test-quant-f32-matmul PRIVATE ggml ggml-cpu)

add_executable(test-gemma-decision-diff tests/test_gemma_decision_diff.cpp)
target_link_libraries(test-gemma-decision-diff PRIVATE crispembed)
add_executable(test-gemma-decision-schema tests/test_gemma_decision_schema.cpp)
target_link_libraries(test-gemma-decision-schema PRIVATE crispembed-core)
target_include_directories(test-gemma-decision-schema PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_executable(test-gemma-decision-activation tests/test_gemma_decision_activation.cpp)
target_link_libraries(test-gemma-decision-activation PRIVATE crispembed)

add_executable(test-qwen35-attention tests/test_qwen35_attention.cpp)
target_link_libraries(test-qwen35-attention PRIVATE crispembed-qwen35-delta crispembed-core)

add_executable(test-qwen35-position tests/test_qwen35_position.cpp)
target_link_libraries(test-qwen35-position PRIVATE crispembed-qwen35-delta crispembed-core)

add_executable(test-ovisocr2-ffn tests/test_ovisocr2_ffn.cpp)
# The test textually includes src/qwen35_decoder.cpp with every external entry point
# renamed, so it links the full library without duplicate definitions.
target_link_libraries(test-ovisocr2-ffn PRIVATE crispembed-qwen35-delta crispembed-core)

add_executable(test-qwen35-decoder tests/test_qwen35_decoder.cpp)
target_link_libraries(test-qwen35-decoder PRIVATE crispembed-qwen35-delta crispembed-core)

add_executable(test-qwen35-vision tests/test_qwen35_vision.cpp)
target_link_libraries(test-qwen35-vision PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-qwen35-norm-statistics tests/test_qwen35_norm_statistics.cpp)
target_link_libraries(test-qwen35-norm-statistics PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-qwen35-attention-trace tests/test_qwen35_attention_trace.cpp)
target_link_libraries(test-qwen35-attention-trace PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-qwen35-vision-blocks tests/test_qwen35_vision_blocks.cpp)
target_link_libraries(test-qwen35-vision-blocks PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-qwen35-vision-ops tests/test_qwen35_vision_ops.cpp)
target_compile_definitions(test-qwen35-vision-ops PRIVATE CRISPEMBED_QWEN35_VISION_OPERATOR_AUDIT=1)
target_link_libraries(test-qwen35-vision-ops PRIVATE crispembed-qwen35-delta crispembed-core)

add_executable(test-qwen35-positions tests/test_qwen35_positions.cpp)
target_link_libraries(test-qwen35-positions PRIVATE crispembed-qwen35-delta)

add_executable(test-qwen35-frontend tests/test_qwen35_frontend.cpp)
target_link_libraries(test-qwen35-frontend PRIVATE crispembed-qwen35-delta crispembed-core)

add_executable(test-ovisocr2-frontend tests/test_ovisocr2_frontend.cpp)
target_link_libraries(test-ovisocr2-frontend PRIVATE crispembed-qwen35-delta crispembed-core)

add_executable(test-ovisocr2-context tests/test_ovisocr2_context.cpp)
target_link_libraries(test-ovisocr2-context PRIVATE crispembed-qwen35-delta crispembed-core)

add_executable(test-ovisocr3-context tests/test_ovisocr3_context.cpp)
target_link_libraries(test-ovisocr3-context PRIVATE crispembed-qwen35-delta crispembed-core)

add_executable(test-ovisocr3-quality-run tests/ovisocr3_quality_run.cpp)
target_link_libraries(test-ovisocr3-quality-run PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-ovisocr2-quality-run tests/ovisocr2_quality_run.cpp)
target_link_libraries(test-ovisocr2-quality-run PRIVATE crispembed-qwen35-delta crispembed-core)
add_executable(test-ovisocr2-image-bounds tests/test_ovisocr2_image_bounds.cpp)
target_link_libraries(test-ovisocr2-image-bounds PRIVATE crispembed-qwen35-delta crispembed-core)
endif() # NOT iOS: standalone validation executables

# Test-only UltraBERT prepared-ID graph; no production architecture dispatch.
add_executable(test-ultrabert-api tests/test_ultrabert_api.cpp)
target_link_libraries(test-ultrabert-api PRIVATE crispembed)

add_executable(test-ultrabert-truncation tests/test_ultrabert_truncation.cpp src/tokenizer_bpe.cpp)
target_link_libraries(test-ultrabert-truncation PRIVATE crispembed-core)

add_executable(test-ultrabert-profile tests/test_ultrabert_profile.cpp src/tokenizer_bpe.cpp)
target_link_libraries(test-ultrabert-profile PRIVATE crispembed-core)

add_executable(test-ultrabert-diff tests/test_ultrabert_diff.cpp)
target_link_libraries(test-ultrabert-diff PRIVATE crispembed-core ggml-cpu)

add_executable(test-ultrabert-rope tests/test_ultrabert_rope.cpp)
target_link_libraries(test-ultrabert-rope PRIVATE crispembed-core ggml-cpu)

# Test-only UltraBERT tokenizer profile; does not add production dispatch.
add_executable(test-ultrabert-tokenizer tests/test_ultrabert_tokenizer.cpp src/tokenizer_bpe.cpp)
target_link_libraries(test-ultrabert-tokenizer PRIVATE crispembed-core)

# Test-only instrumentation uses the real root backend and CPU compiler flags.
# The ordinary library and consumers never receive observer macros or sources.
option(CRISPEMBED_SYSTEMONE_AUDIT "Build test-only SystemOne root-profile observers" OFF)
if(CRISPEMBED_SYSTEMONE_AUDIT)
    add_subdirectory(tests/systemone)
endif()
