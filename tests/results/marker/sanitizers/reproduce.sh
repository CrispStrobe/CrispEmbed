#!/bin/sh
# Run from the repository root. Requires GCC with ASAN/UBSAN runtimes.
# No models, PDFium, network access or Python dependencies are needed.
set -eu
mkdir -p build-marker-sanitizers
g++ -std=c++17 -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined -fno-sanitize-recover=all \
    -Wall -Wextra -Wpedantic -I src \
    src/marker_document.cpp tests/test-marker-document.cpp \
    -o build-marker-sanitizers/test-marker-document-asan-ubsan
# clean_exit/_Exit deliberately bypasses atexit handlers; no leak claim.
ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:strict_string_checks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    ./build-marker-sanitizers/test-marker-document-asan-ubsan
