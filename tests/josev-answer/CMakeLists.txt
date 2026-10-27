cmake_minimum_required(VERSION 3.20)
project(crispembed_josev_answer LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
get_filename_component(CRISPEMBED_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Header-only encoded_text definitions; no frontend implementation, ggml, model,
# tokenizer, or backend dependency is compiled/linked into this lane.
add_library(josev-answer-plain STATIC "${CRISPEMBED_ROOT}/src/josev2b_answer_serializer.cpp"
                                     "${CRISPEMBED_ROOT}/src/core/shortest_double.c")
add_library(josev-answer-tested STATIC "${CRISPEMBED_ROOT}/src/josev2b_answer_serializer.cpp"
                                      "${CRISPEMBED_ROOT}/src/core/shortest_double.c")
target_compile_definitions(josev-answer-tested PRIVATE CRISPEMBED_JOSEV_ANSWER_TESTING)
foreach(target josev-answer-plain josev-answer-tested)
  target_include_directories(${target} PUBLIC "${CRISPEMBED_ROOT}/src")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -ffp-contract=off)
  endif()
endforeach()
add_executable(test-josev-answer "${CRISPEMBED_ROOT}/tests/test_josev2b_answer.cpp")
target_compile_definitions(test-josev-answer PRIVATE CRISPEMBED_JOSEV_ANSWER_TESTING)
target_link_libraries(test-josev-answer PRIVATE josev-answer-tested)

add_executable(test-josev-answer-plain "${CRISPEMBED_ROOT}/tests/test_josev2b_answer.cpp")
target_link_libraries(test-josev-answer-plain PRIVATE josev-answer-plain)
