# CrispEmbed Flutter FFI plugin — Linux
#
# Bundles a pre-built libcrispembed.so with the Flutter application.
# CI builds the shared library and places it at linux/lib/libcrispembed.so.
# See .github/workflows/build.yml for the build matrix.

cmake_minimum_required(VERSION 3.14)

# Glob picks up the full symlink chain (.so → .so.1 → .so.1.0.0) so the
# runtime linker can resolve the soname.  Only search the controlled staging
# directory — CI / the developer copies the library here explicitly.
file(GLOB CRISPEMBED_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/lib/libcrispembed.so*")

if(NOT CRISPEMBED_LIBS)
  message(WARNING
    "crispembed: pre-built libcrispembed.so not found in ${CMAKE_CURRENT_SOURCE_DIR}/lib/. "
    "The plugin will not bundle the native library. "
    "Run the CI build or build manually:\n"
    "  cd <CrispEmbed> && cmake -B build -DCRISPEMBED_BUILD_SHARED=ON && cmake --build build\n"
    "  cp build/libcrispembed.so* flutter/crispembed/linux/lib/")
  set(crispembed_bundled_libraries "" PARENT_SCOPE)
  return()
endif()

# Tell Flutter to bundle the shared libraries (incl. symlinks) with the application.
set(crispembed_bundled_libraries "${CRISPEMBED_LIBS}" PARENT_SCOPE)
