# CrispEmbed Flutter FFI plugin — Linux
#
# Bundles a pre-built libcrispembed.so with the Flutter application.
# CI builds the shared library and places it at linux/lib/libcrispembed.so.
# See .github/workflows/build.yml for the build matrix.

cmake_minimum_required(VERSION 3.14)

set(CRISPEMBED_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib/libcrispembed.so")

if(NOT EXISTS "${CRISPEMBED_LIB}")
  message(WARNING
    "crispembed: pre-built libcrispembed.so not found at ${CRISPEMBED_LIB}. "
    "The plugin will not bundle the native library. "
    "Run the CI build or build manually:\n"
    "  cd <CrispEmbed> && cmake -B build -DCRISPEMBED_BUILD_SHARED=ON && cmake --build build\n"
    "  cp build/libcrispembed.so flutter/crispembed/linux/lib/")
  set(crispembed_bundled_libraries "" PARENT_SCOPE)
  return()
endif()

# Tell Flutter to bundle the shared library with the application.
set(crispembed_bundled_libraries "${CRISPEMBED_LIB}" PARENT_SCOPE)
