cmake_minimum_required(VERSION 3.20)
project(crispembed_moedern_sdpa_diff LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(CRISPEMBED_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Header-only actual Ref. No model runtime, ggml backend or root CMake build.
add_executable(moedern-sdpa-diff diff.cpp)
target_include_directories(moedern-sdpa-diff PRIVATE "${CRISPEMBED_ROOT}/src")
