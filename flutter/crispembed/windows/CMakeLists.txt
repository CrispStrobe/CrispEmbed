# CrispEmbed Flutter FFI plugin — Windows
#
# Bundles a pre-built crispembed.dll with the Flutter application.
# CI builds the DLL and places it at windows/lib/crispembed.dll.
# See .github/workflows/build.yml for the build matrix.

cmake_minimum_required(VERSION 3.14)

set(CRISPEMBED_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib/crispembed.dll")

if(NOT EXISTS "${CRISPEMBED_LIB}")
  message(WARNING
    "crispembed: pre-built crispembed.dll not found at ${CRISPEMBED_LIB}. "
    "The plugin will not bundle the native library. "
    "Run the CI build or build manually:\n"
    "  cd <CrispEmbed> && cmake -S . -B build -DCRISPEMBED_BUILD_SHARED=ON && cmake --build build --config Release\n"
    "  copy build\\Release\\crispembed.dll flutter\\crispembed\\windows\\lib\\")
  set(crispembed_bundled_libraries "" PARENT_SCOPE)
  return()
endif()

# Tell Flutter to bundle the DLL with the application.
set(crispembed_bundled_libraries "${CRISPEMBED_LIB}" PARENT_SCOPE)
