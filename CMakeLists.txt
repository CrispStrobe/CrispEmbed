cmake_minimum_required(VERSION 3.14)
project(crispembed LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ggml (submodule)
add_subdirectory(ggml)

# Core GGUF loader library
add_library(crispembed-core STATIC
    src/core/gguf_loader.cpp
)
target_include_directories(crispembed-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_link_libraries(crispembed-core PUBLIC ggml)

# CrispEmbed library
add_library(crispembed STATIC
    src/crispembed.cpp
    src/tokenizer.cpp
)
target_include_directories(crispembed PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_link_libraries(crispembed PUBLIC
    crispembed-core
    ggml
    ggml-base
    ggml-cpu
    ${CMAKE_THREAD_LIBS_INIT}
)

find_package(Threads REQUIRED)
target_link_libraries(crispembed PUBLIC Threads::Threads)

# CLI binary
add_executable(crispembed-cli
    examples/cli/main.cpp
)
target_link_libraries(crispembed-cli PRIVATE crispembed)
set_target_properties(crispembed-cli PROPERTIES OUTPUT_NAME crispembed)
install(TARGETS crispembed-cli RUNTIME)
