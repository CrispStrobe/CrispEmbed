#include "qwen35_profile.h"
#include "core/env_gate.h"
#include "ggml.h"
#include <chrono>
#include <cstdio>

namespace qwen35_ocr {
namespace {
thread_local profile_record * current = nullptr;
thread_local profile_phase phase_now = profile_phase::other;
uint64_t now() noexcept {
    return uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
const char * phase_names[] = { "other", "vision", "prefill", "decode" };
const char * event_names[] = { "request_file",  "request_raw",     "file_decode",    "preprocess",
                               "prompt",        "vision",          "embedding",      "prefill",
                               "selection",     "cached_prepare",  "cached_decode",  "text_decode",
                               "project_total", "project_setup",   "project_input",  "project_materialize",
                               "project_graph", "project_compute", "project_output", "project_cleanup",
                               "source_buffer", "source_read",     "source_convert", "source_validate" };
static_assert(sizeof(event_names) / sizeof(*event_names) == size_t(profile_event::count));
static_assert(sizeof(phase_names) / sizeof(*phase_names) == size_t(profile_phase::count));
} // namespace
bool profile_active() noexcept {
    return current != nullptr;
}
profile_capture::profile_capture(profile_record & record) {
    if (!current && core_env::on("CRISPEMBED_QWEN35_PROFILE")) {
        record = {};
        current = &record;
        phase_now = profile_phase::other;
        enabled_ = true;
    }
}
profile_capture::~profile_capture() {
    if (enabled_) {
        current = nullptr;
        phase_now = profile_phase::other;
    }
}
profile_timer::profile_timer(profile_event event, profile_phase phase) noexcept
    : record_(current), event_(event), phase_(phase_now), previous_(phase_now) {
    if (!record_) return;
    if (phase != profile_phase::count) phase_ = phase_now = phase;
    started_ = now();
}
void profile_timer::next(profile_event event, profile_phase phase) noexcept {
    if (!record_) return;
    const auto ended = now();
    auto & value = record_->events[size_t(phase_)][size_t(event_)];
    ++value.calls;
    value.nanoseconds += ended - started_;
    event_ = event;
    if (phase != profile_phase::count) phase_ = phase_now = phase;
    started_ = ended;
}
void profile_timer::stop() noexcept {
    if (!record_) return;
    const auto ended = now();
    auto & value = record_->events[size_t(phase_)][size_t(event_)];
    ++value.calls;
    value.nanoseconds += ended - started_;
    phase_now = previous_;
    record_ = nullptr;
}
profile_matrix_scope::profile_matrix_scope(const ggml_tensor * tensor) noexcept
    : record_(current), tensor_(tensor), phase_(phase_now) {
    if (record_) started_ = now();
}
profile_matrix_scope::~profile_matrix_scope() {
    if (!record_) return;
    const auto elapsed = now() - started_;
    try {
        const auto key = std::make_pair(phase_, std::string(ggml_get_name(tensor_)));
        if (record_->matrices.size() >= 2048 && record_->matrices.find(key) == record_->matrices.end()) {
            record_->dropped = true;
            return;
        }
        auto & value = record_->matrices[key];
        ++value.calls;
        value.source_bytes += ggml_nbytes(tensor_);
        value.f32_bytes += uint64_t(ggml_nelements(tensor_)) * sizeof(float);
        value.materialize_ns += elapsed;
    } catch (...) {
        // Profiling must not turn an otherwise successful inference into an
        // allocation exception. A dropped record invalidates the measurement.
        record_->dropped = true;
    }
}
void profile_print(const profile_record & record, bool enabled) {
    size_t events = 0;
    for (size_t p = 0; p < size_t(profile_phase::count); ++p)
        for (size_t e = 0; e < size_t(profile_event::count); ++e) {
            const auto & value = record.events[p][e];
            if (!value.calls) continue;
            ++events;
            std::printf("PROFILE phase=%s event=%s calls=%llu nanoseconds=%llu\n", phase_names[p], event_names[e],
                        (unsigned long long)value.calls, (unsigned long long)value.nanoseconds);
        }
    for (const auto & item : record.matrices) {
        const auto & value = item.second;
        std::printf(
            "PROFILE_MATRIX phase=%s tensor=%s calls=%llu source_bytes=%llu f32_bytes=%llu materialize_ns=%llu\n",
            phase_names[size_t(item.first.first)], item.first.second.c_str(), (unsigned long long)value.calls,
            (unsigned long long)value.source_bytes, (unsigned long long)value.f32_bytes,
            (unsigned long long)value.materialize_ns);
    }
    std::printf("PROFILE_COMPLETE active=%d dropped=%d events=%zu matrices=%zu\n", enabled, record.dropped, events,
                record.matrices.size());
}
} // namespace qwen35_ocr
