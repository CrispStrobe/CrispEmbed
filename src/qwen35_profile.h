#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
struct ggml_tensor;
namespace qwen35_ocr {
// Internal diagnostic counters only: no model/reference values or inference cache.
// A caller-owned session requires CRISPEMBED_QWEN35_PROFILE=1. No clocks or
// allocations are used by inactive timers; sessions are isolated by thread.
enum class profile_phase { other, vision, prefill, decode, count };
enum class profile_event {
    request_file,
    request_raw,
    file_decode,
    preprocess,
    prompt,
    vision,
    embedding,
    prefill,
    selection,
    cached_prepare,
    cached_decode,
    text_decode,
    project_total,
    project_setup,
    project_input,
    project_materialize,
    project_graph,
    project_compute,
    project_output,
    project_cleanup,
    source_buffer,
    source_read,
    source_convert,
    source_validate,
    count
};
struct profile_counter {
    uint64_t calls = 0, nanoseconds = 0;
};
struct profile_matrix_counter {
    uint64_t calls = 0, source_bytes = 0, f32_bytes = 0, materialize_ns = 0;
};
struct profile_record {
    std::array<std::array<profile_counter, size_t(profile_event::count)>, size_t(profile_phase::count)> events{};
    std::map<std::pair<profile_phase, std::string>, profile_matrix_counter> matrices;
    bool dropped = false;
};
bool profile_active() noexcept;
class profile_capture {
    bool enabled_ = false;

public:
    explicit profile_capture(profile_record & record);
    ~profile_capture();
    profile_capture(const profile_capture &) = delete;
    profile_capture & operator=(const profile_capture &) = delete;
    bool enabled() const { return enabled_; }
};
class profile_timer {
    profile_record * record_ = nullptr;
    profile_event event_;
    profile_phase phase_, previous_;
    uint64_t started_ = 0;

public:
    explicit profile_timer(profile_event event, profile_phase phase = profile_phase::count) noexcept;
    ~profile_timer() { stop(); }
    profile_timer(const profile_timer &) = delete;
    profile_timer & operator=(const profile_timer &) = delete;
    void next(profile_event event, profile_phase phase = profile_phase::count) noexcept;
    void stop() noexcept;
};
class profile_matrix_scope {
    profile_record * record_ = nullptr;
    const ggml_tensor * tensor_ = nullptr;
    profile_phase phase_;
    uint64_t started_ = 0;

public:
    explicit profile_matrix_scope(const ggml_tensor * tensor) noexcept;
    ~profile_matrix_scope();
    profile_matrix_scope(const profile_matrix_scope &) = delete;
    profile_matrix_scope & operator=(const profile_matrix_scope &) = delete;
};
void profile_print(const profile_record & record, bool enabled);
} // namespace qwen35_ocr
