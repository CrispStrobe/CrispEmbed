cmake_minimum_required(VERSION 3.16)
project(crispembed_ar_morph LANGUAGES C CXX)

# Closed standalone target for the native Camel Morph / camel_tools analyzer:
# no ggml, no model weights, no third-party libraries. The database is CC BY
# 4.0 data downloaded separately; the ported camel_tools code is MIT.
get_filename_component(CRISPEMBED_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CRISPEMBED_AR_MORPH_STATIC (default OFF) builds a static library for hosts
# that link the runtime into a signed app (iOS).
option(CRISPEMBED_AR_MORPH_STATIC "Build crispembed-ar-morph as a static library" OFF)
if(CRISPEMBED_AR_MORPH_STATIC)
    set(_ar_morph_kind STATIC)
else()
    set(_ar_morph_kind SHARED)
endif()
add_library(crispembed-ar-morph ${_ar_morph_kind}
    "${CRISPEMBED_ROOT}/src/arabic_morph.cpp"
    "${CRISPEMBED_ROOT}/src/core/shortest_double.c")
target_include_directories(crispembed-ar-morph PUBLIC "${CRISPEMBED_ROOT}/src")
target_include_directories(crispembed-ar-morph PRIVATE "${CRISPEMBED_ROOT}/src/core/third_party")
if(CRISPEMBED_AR_MORPH_STATIC)
    target_compile_definitions(crispembed-ar-morph PRIVATE CRISPEMBED_AR_MORPH_BUILD)
else()
    target_compile_definitions(crispembed-ar-morph PRIVATE CRISPEMBED_AR_MORPH_BUILD PUBLIC CRISPEMBED_AR_MORPH_SHARED)
endif()
if(MSVC)
    target_compile_options(crispembed-ar-morph PRIVATE /W4 /utf-8)
else()
    target_compile_options(crispembed-ar-morph PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(crispembed-ar-morph-cli "${CMAKE_CURRENT_LIST_DIR}/ar_morph_cli.cpp")
target_link_libraries(crispembed-ar-morph-cli PRIVATE crispembed-ar-morph)
set_target_properties(crispembed-ar-morph-cli PROPERTIES OUTPUT_NAME crispembed-ar-morph)
if(MSVC)
    target_compile_options(crispembed-ar-morph-cli PRIVATE /utf-8)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test-arabic-morph "${CRISPEMBED_ROOT}/tests/test_arabic_morph.cpp")
    target_link_libraries(test-arabic-morph PRIVATE crispembed-ar-morph)
    if(MSVC)
        target_compile_options(test-arabic-morph PRIVATE /utf-8)
    endif()
    add_test(NAME arabic-morph-contract COMMAND test-arabic-morph)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-tmp")
    set_tests_properties(arabic-morph-contract PROPERTIES
        ENVIRONMENT "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/test-tmp;TEMP=${CMAKE_CURRENT_BINARY_DIR}/test-tmp")
endif()

install(TARGETS crispembed-ar-morph LIBRARY DESTINATION lib RUNTIME DESTINATION bin ARCHIVE DESTINATION lib)
install(FILES "${CRISPEMBED_ROOT}/src/arabic_morph.h" DESTINATION include)
install(FILES "${CRISPEMBED_ROOT}/LICENSE"
        DESTINATION share/licenses/crispembed-ar-morph RENAME CrispEmbed-LICENSE)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/THIRD_PARTY_NOTICES"
        DESTINATION share/licenses/crispembed-ar-morph)
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/notices/"
        DESTINATION share/licenses/crispembed-ar-morph/notices)
