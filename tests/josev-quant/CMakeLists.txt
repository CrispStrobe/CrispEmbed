cmake_minimum_required(VERSION 3.20)
project(crispembed_josev_quant_controls LANGUAGES C CXX)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../josev-packed" packed)
# The shared qwen35 TUs in josev-packed-plain call the LightOn4B physical helpers.
# Without CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI these compile to the closed stubs the
# main library uses; linking them here leaves the (frozen) josev-packed project as is.
get_filename_component(CRISPEMBED_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_executable(josev-quant-controls controls.cpp
    "${CRISPEMBED_ROOT}/src/lighton4b_physical_gdn_source_ops.cpp"
    "${CRISPEMBED_ROOT}/src/lighton4b_physical_q6_tiled.cpp"
    "${CRISPEMBED_ROOT}/src/lighton4b_physical_f16_attention.cpp")
target_link_libraries(josev-quant-controls PRIVATE josev-packed-plain)
