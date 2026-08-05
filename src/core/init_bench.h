#pragma once
// init_bench.h — per-component one-shot init profile, gated on CRISPEMBED_INIT_BENCH=1.
//
// T18: a one-shot `crispembed -m model.gguf --json "text"` pays a fixed init
// cost that dwarfs the ~6-12 ms of actual compute. This is the instrument that
// says WHERE the cost is, so a future session does not have to re-derive it by
// bisecting the loader. It is a measurement tool, not a feature: everything is
// off unless CRISPEMBED_INIT_BENCH is set, and the only cost when off is one
// getenv per process plus a monotonic clock read per mark.
//
// Usage:
//     core_initbench::timer t("load_model");
//     ... work ...
//     t.mark("gguf_meta");
//     ... work ...
//     t.mark("tokenizer");
//     // destructor prints the total
//
// Output goes to stderr, one line per component:
//     crispembed init-bench: load_model/gguf_meta            123.4 ms
//     crispembed init-bench: load_model TOTAL                456.7 ms

#include "env_gate.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace core_initbench {

inline bool enabled() {
    // Shared value-parsed semantics (set and not "0" => on); see core/env_gate.h.
    static const bool on = core_env::on("CRISPEMBED_INIT_BENCH");
    return on;
}

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Process start, captured the first time anything asks. Lets a mark report
// wall-clock-since-first-timer as well as its own component slice.
inline double origin_ms() {
    static const double t0 = now_ms();
    return t0;
}

class timer {
public:
    explicit timer(const char * scope) : scope_(scope ? scope : "") {
        if (!enabled()) return;
        origin_ms(); // pin the origin at the first timer construction
        t_start_ = now_ms();
        t_last_ = t_start_;
    }

    // Close out the component named `label` (time since the previous mark).
    void mark(const char * label) {
        if (!enabled()) return;
        const double now = now_ms();
        fprintf(stderr, "crispembed init-bench: %-40s %8.1f ms   (+%.1f ms wall)\n",
                (scope_ + "/" + (label ? label : "?")).c_str(), now - t_last_, now - origin_ms());
        t_last_ = now;
    }

    // Total for the scope, printed explicitly (the destructor also prints if
    // this was never called).
    void total() {
        if (!enabled() || printed_total_) return;
        printed_total_ = true;
        const double now = now_ms();
        fprintf(stderr, "crispembed init-bench: %-40s %8.1f ms   (+%.1f ms wall)\n", (scope_ + " TOTAL").c_str(),
                now - t_start_, now - origin_ms());
    }

    ~timer() { total(); }

private:
    std::string scope_;
    double t_start_ = 0.0;
    double t_last_ = 0.0;
    bool printed_total_ = false;
};

} // namespace core_initbench
