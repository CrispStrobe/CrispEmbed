# Conservative source/configuration identity for resumable extraction. This is
# generated from packaged source bytes, so a Git checkout is not required.
file(GLOB_RECURSE _marker_identity_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/*")
list(APPEND _marker_identity_sources "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MarkerIdentity.cmake")
list(SORT _marker_identity_sources)
set(_marker_identity "crispembed.marker-build/1\n")
foreach(_marker_file IN LISTS _marker_identity_sources)
    if(NOT IS_DIRECTORY "${_marker_file}")
        file(SHA256 "${_marker_file}" _marker_digest)
        file(RELATIVE_PATH _marker_relative "${CMAKE_CURRENT_SOURCE_DIR}" "${_marker_file}")
        string(APPEND _marker_identity "${_marker_relative}=${_marker_digest}\n")
    endif()
endforeach()
# Reconfigure when an existing source changes, not only when the glob changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_marker_identity_sources})
get_cmake_property(_marker_variables VARIABLES)
list(SORT _marker_variables)
foreach(_marker_variable IN LISTS _marker_variables)
    if(_marker_variable MATCHES "^(GGML_|CRISPEMBED_NATIVE$|CMAKE_C_FLAGS|CMAKE_CXX_FLAGS|CMAKE_BUILD_TYPE$|CMAKE_SYSTEM_NAME$|CMAKE_SYSTEM_PROCESSOR$|CMAKE_C_COMPILER_ID$|CMAKE_C_COMPILER_VERSION$|CMAKE_CXX_COMPILER_ID$|CMAKE_CXX_COMPILER_VERSION$)")
        string(APPEND _marker_identity "${_marker_variable}=${${_marker_variable}}\n")
    endif()
endforeach()
string(SHA256 CRISPEMBED_MARKER_BUILD_ID "${_marker_identity}")
unset(_marker_identity)
unset(_marker_identity_sources)
unset(_marker_variables)
