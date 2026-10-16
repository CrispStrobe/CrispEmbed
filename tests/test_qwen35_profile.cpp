#include "core/clean_exit.h"
#include "qwen35_profile.h"
#include "qwen35_ocr.h"
#include "ggml.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

using namespace qwen35_ocr;
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
static bool empty(const profile_record & r) {
    if (!r.matrices.empty() || r.dropped) return false;
    for (const auto & phase : r.events)
        for (const auto & event : phase)
            if (event.calls || event.nanoseconds) return false;
    return true;
}
static int controls() {
    for (const char * value : { "", "0" }) {
        gate("CRISPEMBED_QWEN35_PROFILE", value);
        profile_record r;
        {
            profile_capture capture(r);
            profile_timer timer(profile_event::project_setup, profile_phase::vision);
            if (capture.enabled() || profile_active()) return 1;
        }
        if (!empty(r)) return 1;
    }
    gate("CRISPEMBED_QWEN35_PROFILE", "1");
    profile_record r, nested, independent;
    {
        profile_capture capture(r);
        if (!capture.enabled() || !profile_active()) return 1;
        {
            profile_timer outer(profile_event::vision, profile_phase::vision);
            { profile_timer inner(profile_event::project_compute); }
            outer.next(profile_event::prefill, profile_phase::prefill);
            { profile_timer inner(profile_event::project_compute); }
            outer.stop();
            outer.stop(); // An explicitly stopped timer must not double count.
        }
        { profile_timer restored(profile_event::selection); }
        {
            profile_capture rejected(nested);
            if (rejected.enabled()) return 1;
            profile_timer parent(profile_event::embedding);
        }
        std::thread other([&] {
            if (profile_active()) return;
            profile_capture own(independent);
            profile_timer timer(profile_event::text_decode);
        });
        other.join();
        ggml_tensor tensor{};
        tensor.type = GGML_TYPE_F32;
        for (int i = 0; i < 4; ++i) tensor.ne[i] = 1;
        tensor.ne[0] = 2;
        tensor.nb[0] = 4;
        for (int i = 1; i < 4; ++i) tensor.nb[i] = 8;
        ggml_set_name(&tensor, "test.matrix");
        { profile_matrix_scope first(&tensor); }
        { profile_matrix_scope second(&tensor); }
        const auto & matrix = r.matrices.at({ profile_phase::other, "test.matrix" });
        if (matrix.calls != 2 || matrix.source_bytes != 16 || matrix.f32_bytes != 16) return 1;
    }
    const auto calls = [&](profile_phase p, profile_event e) { return r.events[size_t(p)][size_t(e)].calls; };
    if (profile_active() || !empty(nested) || r.dropped || calls(profile_phase::vision, profile_event::vision) != 1 ||
        calls(profile_phase::vision, profile_event::project_compute) != 1 ||
        calls(profile_phase::prefill, profile_event::prefill) != 1 ||
        calls(profile_phase::prefill, profile_event::project_compute) != 1 ||
        calls(profile_phase::other, profile_event::selection) != 1 ||
        calls(profile_phase::other, profile_event::embedding) != 1 ||
        independent.events[size_t(profile_phase::other)][size_t(profile_event::text_decode)].calls != 1)
        return 1;
    gate("CRISPEMBED_QWEN35_PROFILE", "0");
    std::puts(
        "PROFILE_CONTROLS pass=1 literal0=1 inactive_empty=1 nested=1 phase_restore=1 thread_local=1 byte_counts=1");
    return 0;
}
static int benchmark(int argc, char ** argv) {
    if (argc == 1) return controls();
    if (argc != 4) return 2;
    char * end = nullptr;
    const long cap = std::strtol(argv[3], &end, 10);
    if (!end || *end || cap < 1 || cap > 512) return 2;
    gate("CRISPEMBED_QWEN35_OCR", "1");
    const auto load_started = std::chrono::steady_clock::now();
    using owner = std::unique_ptr<qwen35_ocr_context, decltype(&qwen35_ocr_free)>;
    owner context(qwen35_ocr_init(argv[1], 4), qwen35_ocr_free);
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_started).count();
    if (!context || !qwen35_ocr_set_max_tokens(context.get(), int(cap))) return 2;
    profile_record record;
    bool active = false;
    int length = 0;
    const char * text = nullptr;
    double ms = 0;
    {
        profile_capture capture(record);
        active = capture.enabled();
        const auto started = std::chrono::steady_clock::now();
        text = qwen35_ocr_recognize_file(context.get(), argv[2], "", &length);
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    }
    int count = 0, probability_count = 0;
    const int * ids = qwen35_ocr_token_ids(context.get(), &count);
    const float * probabilities = qwen35_ocr_confidences(context.get(), &probability_count);
    if (!text || !ids || !probabilities || count < 1 || count != probability_count || length < 0) return 2;
    std::printf("BENCH ids:");
    for (int i = 0; i < count; ++i) std::printf(" %d", ids[i]);
    std::printf("\nBENCH confidences:");
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(probabilities[i]) || probabilities[i] < 0 || probabilities[i] > 1) return 2;
        std::printf(" %.9g", probabilities[i]);
    }
    std::printf("\nBENCH utf8_hex:");
    for (int i = 0; i < length; ++i) std::printf("%02x", static_cast<unsigned char>(text[i]));
    std::printf("\nBENCH stop=%d mean_confidence=%.9g bytes=%d load_ms=%.17g recognize_ms=%.17g\n",
                qwen35_ocr_last_stop(context.get()), qwen35_ocr_mean_confidence(context.get()), length, load_ms, ms);
    profile_print(record, active);
    return record.dropped || (!active && !empty(record)) ? 1 : 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(benchmark(argc, argv));
}
