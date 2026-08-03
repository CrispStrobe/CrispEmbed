# CrispEmbedC2pa.cmake — optional C2PA (Content Credentials) support.
#
# Absence is normal and must stay silent-but-stated: without the library,
# images are still emitted as PNG with an iTXt provenance chunk, which is the
# Art. 50(2) machine-readable marking. C2PA adds a signed manifest on top, and
# only when the operator also supplies a certificate — we ship no key. See
# src/core/image_out.h for why.
#
# Layout matches the c2pa-rs prebuilt release archive:
#   third_party/c2pa/include/c2pa.h
#   third_party/c2pa/lib/libc2pa_c.{dylib,so}  (c2pa_c.dll on Windows)
#
# -DCRISPEMBED_C2PA_FETCH=ON downloads it at configure time. Best-effort by
# design: a failed download leaves C2PA off rather than failing the build,
# because a green build without Content Credentials is the correct outcome for
# someone building offline.

if (DEFINED _CRISPEMBED_C2PA_INCLUDED)
    return()
endif()
set(_CRISPEMBED_C2PA_INCLUDED ON)

option(CRISPEMBED_C2PA_FETCH "Download the prebuilt c2pa-rs native lib if absent" OFF)
set(CRISPEMBED_C2PA_VERSION "0.89.3" CACHE STRING "c2pa-rs prebuilt version")

set(_c2pa_root "${CMAKE_SOURCE_DIR}/third_party/c2pa")

if (CRISPEMBED_C2PA_FETCH AND NOT EXISTS "${_c2pa_root}/include/c2pa.h")
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _c2pa_arch)
    if (APPLE)
        if (_c2pa_arch MATCHES "arm64|aarch64")
            set(_c2pa_triple "aarch64-apple-darwin")
        else()
            set(_c2pa_triple "x86_64-apple-darwin")
        endif()
    elseif (UNIX)
        if (_c2pa_arch MATCHES "arm64|aarch64")
            set(_c2pa_triple "aarch64-unknown-linux-gnu")
        else()
            set(_c2pa_triple "x86_64-unknown-linux-gnu")
        endif()
    elseif (WIN32)
        set(_c2pa_triple "x86_64-pc-windows-msvc")
    endif()

    if (_c2pa_triple)
        set(_c2pa_url
            "https://github.com/contentauth/c2pa-rs/releases/download/c2pa-v${CRISPEMBED_C2PA_VERSION}/c2pa-v${CRISPEMBED_C2PA_VERSION}-${_c2pa_triple}.zip")
        message(STATUS "C2PA: fetching ${_c2pa_url}")
        file(DOWNLOAD "${_c2pa_url}" "${CMAKE_BINARY_DIR}/c2pa-prebuilt.zip" STATUS _c2pa_dl TIMEOUT 180)
        list(GET _c2pa_dl 0 _c2pa_dl_code)
        if (_c2pa_dl_code EQUAL 0)
            file(MAKE_DIRECTORY "${_c2pa_root}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/c2pa-prebuilt.zip"
                            WORKING_DIRECTORY "${_c2pa_root}" RESULT_VARIABLE _c2pa_unzip)
            if (NOT _c2pa_unzip EQUAL 0)
                message(STATUS "C2PA: unpack failed — continuing without Content Credentials")
            endif()
        else()
            message(STATUS "C2PA: download failed — continuing without Content Credentials")
        endif()
    endif()
endif()

find_path(CRISPEMBED_C2PA_INCLUDE_DIR c2pa.h HINTS "${_c2pa_root}/include" NO_DEFAULT_PATH)
find_library(CRISPEMBED_C2PA_LIBRARY NAMES c2pa_c HINTS "${_c2pa_root}/lib" NO_DEFAULT_PATH)
if (NOT CRISPEMBED_C2PA_INCLUDE_DIR)
    find_path(CRISPEMBED_C2PA_INCLUDE_DIR c2pa.h)
endif()
if (NOT CRISPEMBED_C2PA_LIBRARY)
    find_library(CRISPEMBED_C2PA_LIBRARY NAMES c2pa_c)
endif()

if (CRISPEMBED_C2PA_INCLUDE_DIR AND CRISPEMBED_C2PA_LIBRARY)
    set(CRISPEMBED_HAVE_C2PA ON)
    message(STATUS "C2PA: enabled (${CRISPEMBED_C2PA_LIBRARY})")
else()
    set(CRISPEMBED_HAVE_C2PA OFF)
    message(STATUS "C2PA: not found — images are still PNG with iTXt provenance, just unsigned "
                   "(-DCRISPEMBED_C2PA_FETCH=ON to enable)")
endif()

# Apply to a target that compiles core/image_out.cpp.
function(crispembed_enable_c2pa tgt)
    if (CRISPEMBED_HAVE_C2PA)
        target_include_directories(${tgt} PRIVATE "${CRISPEMBED_C2PA_INCLUDE_DIR}")
        target_link_libraries(${tgt} PRIVATE "${CRISPEMBED_C2PA_LIBRARY}")
        target_compile_definitions(${tgt} PRIVATE CRISPEMBED_HAVE_C2PA)
    endif()
endfunction()
