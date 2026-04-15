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
    src/decoder_embed.cpp
    src/tokenizer.cpp
    src/tokenizer_spm.cpp
    src/tokenizer_bpe.cpp
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
    examples/cli/model_mgr.cpp
)
target_include_directories(crispembed-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli
)
target_link_libraries(crispembed-cli PRIVATE crispembed)
set_target_properties(crispembed-cli PROPERTIES OUTPUT_NAME crispembed)
install(TARGETS crispembed-cli RUNTIME)

# Server binary
add_executable(crispembed-server
    examples/server/server.cpp
)
target_include_directories(crispembed-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/server
)
target_link_libraries(crispembed-server PRIVATE crispembed)
install(TARGETS crispembed-server RUNTIME)

# Quantize tool
add_executable(crispembed-quantize
    tools/quantize.cpp
)
target_include_directories(crispembed-quantize PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_link_libraries(crispembed-quantize PRIVATE ggml ggml-base ggml-cpu)
set_target_properties(crispembed-quantize PROPERTIES OUTPUT_NAME crispembed-quantize)
install(TARGETS crispembed-quantize RUNTIME)

# Shared library for Python bindings
add_library(crispembed-shared SHARED
    src/crispembed.cpp
    src/tokenizer.cpp
    src/tokenizer_spm.cpp
)
target_include_directories(crispembed-shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_link_libraries(crispembed-shared PUBLIC crispembed-core ggml ggml-base ggml-cpu Threads::Threads)
set_target_properties(crispembed-shared PROPERTIES OUTPUT_NAME crispembed)
