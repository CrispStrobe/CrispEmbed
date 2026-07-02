// src/core/clean_exit.h — terminate a one-shot binary without the GPU-device
// teardown crash.
//
// ggml v0.10.0 (submodule 8be60f83) tears down its process-global GPU device in
// a C++ static destructor at exit. If a one-shot binary leaves GPU buffers alive
// (the common "let the OS reclaim it" pattern), that teardown aborts AFTER
// results are printed:
//   - Metal: GGML_ASSERT([rsets->data count]==0) in ggml_metal_device_free
//   - CUDA:  use-after-free of the destroyed device → SIGSEGV / SIGABRT
// The output is already correct and flushed, so the safe, backend-agnostic fix
// is to skip the static-destructor teardown entirely (the same os._exit trick
// downstream already used for the PyTorch-MPS coexistence case).
//
// Use ONLY for one-shot binaries (the CLI, test-*-diff harnesses). Long-lived
// hosts — the server and language bindings — must instead free their contexts
// via crispembed_free on shutdown (they already do), so they can opt into the
// Metal residency cache with CRISPEMBED_METAL_RESIDENCY=1 and still exit cleanly.
//
// See LEARNINGS.md "ggml v0.10.0 Metal regressions" / memory
// ggml-8be60f-sched-teardown-asserts.

#pragma once

#include <cstdio>
#include <cstdlib>

namespace core_util {

// Flush user-visible output, then terminate immediately WITHOUT running static
// destructors or atexit handlers (so ggml's global GPU-device teardown never
// runs). Do any needed per-context frees (crispembed_free, <engine>_free) BEFORE
// calling this if you want them to happen — std::_Exit does not run them.
[[noreturn]] static inline void clean_exit(int rc) {
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(rc);
}

} // namespace core_util
