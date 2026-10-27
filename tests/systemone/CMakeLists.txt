# Opt-in reference instrumentation; never linked into production archives.
set(SYSTEMONE_AUDIT_SOURCES
    ${PROJECT_SOURCE_DIR}/src/josev2b_backbone.cpp
    ${PROJECT_SOURCE_DIR}/src/qwen35_decoder.cpp
    ${PROJECT_SOURCE_DIR}/src/qwen35_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/qwen35_linear.cpp
    ${PROJECT_SOURCE_DIR}/src/qwen35_delta.cpp
    ${PROJECT_SOURCE_DIR}/src/qwen35_norm.cpp
    ${PROJECT_SOURCE_DIR}/src/qwen35_positions.cpp
    ${PROJECT_SOURCE_DIR}/src/qwen35_profile.cpp
    ${PROJECT_SOURCE_DIR}/src/josev2b_composed.cpp
    ${PROJECT_SOURCE_DIR}/src/josev2b_head.cpp
    ${PROJECT_SOURCE_DIR}/src/josev2b_text_frontend.cpp
    ${PROJECT_SOURCE_DIR}/src/josev2b_answer_serializer.cpp)
foreach(variant plain observed)
    add_library(systemone-audit-${variant} STATIC ${SYSTEMONE_AUDIT_SOURCES})
    target_include_directories(systemone-audit-${variant} PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(systemone-audit-${variant} PUBLIC crispembed-core ggml ggml-cpu Threads::Threads)
    crispembed_apply_arch_flags(systemone-audit-${variant})
    add_executable(systemone-audit-${variant}-probe ${PROJECT_SOURCE_DIR}/tests/josev-packed/probe.cpp)
    target_link_libraries(systemone-audit-${variant}-probe PRIVATE systemone-audit-${variant})
endforeach()
target_compile_definitions(systemone-audit-observed PRIVATE CRISPEMBED_JOSEV_COMPOSED_TESTING CRISPEMBED_JOSEV_HEAD_TESTING)
target_compile_definitions(systemone-audit-observed-probe PRIVATE CRISPEMBED_JOSEV_COMPOSED_TESTING)
add_executable(systemone-audit-diff ${PROJECT_SOURCE_DIR}/tests/josev-packed/diff.cpp)
target_include_directories(systemone-audit-diff PRIVATE ${PROJECT_SOURCE_DIR}/src)
