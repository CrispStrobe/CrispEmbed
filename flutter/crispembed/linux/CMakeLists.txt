# CrispEmbed Flutter FFI plugin — Linux
#
# Bundles a pre-built libcrispembed.so with the Flutter application.
# CI builds the shared library and places it at linux/lib/libcrispembed.so.
# See .github/workflows/build.yml for the build matrix.

cmake_minimum_required(VERSION 3.14)

# Include the native library and its staged runtime dependencies, including the
# full soname symlink chains. Shared builds depend on ggml sibling libraries;
# copying only libcrispembed.so leaves an installed application unable to load
# the package. Only search the controlled directory populated by CI/developers.
file(GLOB CRISPEMBED_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.so*")
file(GLOB CRISPEMBED_MAIN_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/lib/libcrispembed.so*")

if(NOT CRISPEMBED_MAIN_LIBS)
  message(WARNING
    "crispembed: pre-built libcrispembed.so not found in ${CMAKE_CURRENT_SOURCE_DIR}/lib/. "
    "The plugin will not bundle the native library. "
    "Run the CI build or build manually:\n"
    "  cd <CrispEmbed> && cmake -B build -DCRISPEMBED_BUILD_SHARED=ON && cmake --build build\n"
    "  cmake --install build --prefix dist\n"
    "  cp dist/lib/*.so* flutter/crispembed/linux/lib/")
  set(crispembed_bundled_libraries "" PARENT_SCOPE)
  return()
endif()

# Tell Flutter to bundle the shared libraries (incl. symlinks) with the application.
set(crispembed_bundled_libraries "${CRISPEMBED_LIBS}" PARENT_SCOPE)
