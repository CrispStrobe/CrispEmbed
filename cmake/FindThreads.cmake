# cmake/FindThreads.cmake — Emscripten stub override.
#
# When CMAKE_MODULE_PATH includes this directory, this file shadows
# CMake's built-in FindThreads.cmake. It creates a no-op Threads::Threads
# target without running any compiler checks, avoiding the -pthread
# requirement that would force SharedArrayBuffer + COOP/COEP headers.
#
# Only activated when EMSCRIPTEN is defined (build-wasm.sh prepends
# CMAKE_MODULE_PATH). Native builds use the standard FindThreads.

if(EMSCRIPTEN)
    set(Threads_FOUND TRUE)
    set(THREADS_FOUND TRUE)
    set(CMAKE_THREAD_LIBS_INIT "")
    set(CMAKE_USE_PTHREADS_INIT 1)

    if(NOT TARGET Threads::Threads)
        add_library(Threads::Threads INTERFACE IMPORTED)
    endif()
else()
    # Fall through to the real FindThreads for native builds.
    # Remove our directory from CMAKE_MODULE_PATH temporarily.
    list(REMOVE_ITEM CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
    find_package(Threads ${Threads_FIND_VERSION}
        ${Threads_FIND_QUIETLY_ARG} ${Threads_FIND_REQUIRED_ARG})
    list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
endif()
