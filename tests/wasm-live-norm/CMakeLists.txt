cmake_minimum_required(VERSION 3.20)
project(wasm_live_norm LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
get_filename_component(ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Reuse the frozen component's actual a06 CPU ISA/precision profile unchanged.
add_subdirectory("${ROOT}/tests/wasm-native-order-norm" frozen-controls)
set_target_properties(norm-candidate-controls norm-candidate-ref PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(live-norm-replay replay.cpp "${ROOT}/src/core/wasm_native_order_norm.cpp")
target_include_directories(live-norm-replay PRIVATE "${ROOT}/src")
target_link_libraries(live-norm-replay PRIVATE ggml ggml-cpu)
target_compile_options(live-norm-replay PRIVATE -ffp-contract=off)
if(EMSCRIPTEN)
 target_compile_options(live-norm-replay PRIVATE -pthread)
 target_link_options(live-norm-replay PRIVATE -pthread -sMODULARIZE=1 -sEXPORT_NAME=LiveNormReplay
  -sENVIRONMENT=web,worker -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=67108864 -sPTHREAD_POOL_SIZE=2
  "-sEXPORTED_FUNCTIONS=['_live_norm_replay','_malloc','_free']"
  "-sEXPORTED_RUNTIME_METHODS=['ccall','HEAPF32','HEAPU8']")
 set_target_properties(live-norm-replay PROPERTIES SUFFIX ".js")
endif()
