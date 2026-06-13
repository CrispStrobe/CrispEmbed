# Install script for directory: /mnt/volume1/CrispEmbed/ggml

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/mnt/volume1/emsdk/upstream/emscripten/cache/sysroot")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/mnt/volume1/CrispEmbed/build-embed-wasm/ggml/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/mnt/volume1/CrispEmbed/build-embed-wasm/ggml/src/libggml.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/mnt/volume1/CrispEmbed/ggml/include/ggml.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-cpu.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-alloc.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-backend.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-blas.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-cann.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-cpp.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-cuda.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-opt.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-metal.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-rpc.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-virtgpu.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-sycl.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-vulkan.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-webgpu.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-zendnn.h"
    "/mnt/volume1/CrispEmbed/ggml/include/ggml-openvino.h"
    "/mnt/volume1/CrispEmbed/ggml/include/gguf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/mnt/volume1/CrispEmbed/build-embed-wasm/ggml/src/libggml-base.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "/mnt/volume1/CrispEmbed/build-embed-wasm/ggml/ggml-config.cmake"
    "/mnt/volume1/CrispEmbed/build-embed-wasm/ggml/ggml-version.cmake"
    )
endif()

