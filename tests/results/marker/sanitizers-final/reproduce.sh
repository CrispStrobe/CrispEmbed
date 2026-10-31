#!/bin/sh
# Run from the repository root. GCC ASAN/UBSAN runtimes are required.
# Final formatted/control-fixed source: 86 model-free document checks.
set -eu
mkdir -p build-marker-sanitizers-final
g++ -std=c++17 -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined -fno-sanitize-recover=all \
    -Wall -Wextra -Wpedantic -I src \
    src/marker_document.cpp tests/test-marker-document.cpp \
    -o build-marker-sanitizers-final/test-marker-document-asan-ubsan
# clean_exit/_Exit bypasses atexit handlers; no leak-freedom claim.
ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:strict_string_checks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    ./build-marker-sanitizers-final/test-marker-document-asan-ubsan
