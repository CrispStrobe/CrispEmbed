#pragma once
#include "qwen35_profile.h"
// Test-runner-only adapter. Existing CRISPEMBED_QWEN35_PROFILE gate and counters
// are reused; inactive sessions do not read clocks or allocate counter entries.
struct lighton4b_diagnostic_profile {
    qwen35_ocr::profile_record record;
    qwen35_ocr::profile_capture capture{ record };
    ~lighton4b_diagnostic_profile() {
        if (capture.enabled()) qwen35_ocr::profile_print(record, true);
    }
};
template <class F>
bool lighton4b_measured(qwen35_ocr::profile_event event, qwen35_ocr::profile_phase phase, F && function) {
    qwen35_ocr::profile_timer timer(event, phase);
    return function();
}
