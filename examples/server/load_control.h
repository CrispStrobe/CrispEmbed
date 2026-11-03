// load_control.h — admission, load gauges and Prometheus text for
// crispembed-server.
//
// Why this exists: every model on this server sits behind one mutex, so a
// burst of concurrent requests queues invisibly inside httplib's thread pool
// and latency grows without bound. Operators had no way to see that queue and
// no way to bound it. Both knobs below are generic (Prometheus scrape, HTTP
// 429) and default off.
//
//   * CRISPEMBED_SERVER_METRICS=1   — GET /metrics, Prometheus text. Default off.
//   * CRISPEMBED_SERVER_MAX_INFLIGHT=N (N >= 1) — POST requests beyond N in
//     flight are rejected with 429 before they read a model. Unset, empty, 0
//     or unparsable = no cap (historical behaviour).
//   * CRISPEMBED_SERVER_RETRY_AFTER=S — add `Retry-After: S` to those 429s.
//     Default: no header. A bare 429 says "overloaded, back off"; a
//     Retry-After additionally names a wait, which callers commonly treat as
//     a fixed pacing quota rather than load. Opt in when that is what you want.
//
// Gauge semantics (exactly what is observed, nothing estimated):
//   running  = requests currently holding at least one model mutex
//   waiting  = admitted POST requests that do not hold one yet (queued on a
//              mutex, reading/parsing, or serialising the response)
// "running" is counted by counted_mutex below, which the server uses for every
// model mutex; a thread-local depth keeps a request that holds two model locks
// from counting twice.
//
// Header-only and free of httplib/ggml so tests/test_server_load_control.cpp
// can exercise it host-only.
#pragma once

#include "core/env_gate.h"
#include "core/env_value.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>

namespace server_load {

struct counters {
    std::atomic<int64_t> inflight{ 0 }; // admitted, not yet finished (POST only)
    std::atomic<int64_t> running{ 0 };  // holding >= 1 model mutex
    std::atomic<int64_t> admitted_total{ 0 };
    std::atomic<int64_t> rejected_total{ 0 }; // 429 at admission
    std::atomic<int64_t> completed_total{ 0 };
    std::atomic<int64_t> duration_us_total{ 0 };       // admitted request wall time
    std::atomic<int64_t> prompt_tokens_total{ 0 };     // as reported in responses
    std::atomic<int64_t> completion_tokens_total{ 0 }; // as reported in responses
};

inline counters & global() {
    static counters c;
    return c;
}

inline int & lock_depth() {
    static thread_local int depth = 0;
    return depth;
}

// Drop-in for std::mutex (BasicLockable + try_lock) that maintains `running`.
class counted_mutex {
public:
    explicit counted_mutex(counters * c = nullptr) : c_(c ? c : &global()) {}
    counted_mutex(const counted_mutex &) = delete;
    counted_mutex & operator=(const counted_mutex &) = delete;

    void lock() {
        m_.lock();
        enter();
    }
    bool try_lock() {
        if (!m_.try_lock()) return false;
        enter();
        return true;
    }
    void unlock() {
        if (--lock_depth() == 0) c_->running.fetch_sub(1, std::memory_order_relaxed);
        m_.unlock();
    }

private:
    void enter() {
        if (lock_depth()++ == 0) c_->running.fetch_add(1, std::memory_order_relaxed);
    }
    std::mutex m_;
    counters * c_;
};

// Parse a non-negative integer env knob; anything else (unset, empty, junk,
// negative, overflow) is `fallback`.
inline int64_t env_int(const char * name, int64_t fallback) {
    const char * e = core_env::value(name); // empty == unset
    if (!e) return fallback;
    errno = 0;
    char * end = nullptr;
    const long long v = std::strtoll(e, &end, 10);
    if (errno != 0 || !end || *end != '\0' || v < 0) return fallback;
    return (int64_t)v;
}

struct config {
    bool metrics = false;     // GET /metrics
    int64_t max_inflight = 0; // 0 = uncapped
    int64_t retry_after = -1; // < 0 = no Retry-After header

    static config from_env() {
        config c;
        c.metrics = core_env::on("CRISPEMBED_SERVER_METRICS");
        c.max_inflight = env_int("CRISPEMBED_SERVER_MAX_INFLIGHT", 0);
        c.retry_after = env_int("CRISPEMBED_SERVER_RETRY_AFTER", -1);
        return c;
    }
};

// Admission. Returns false (and counts a rejection) when a cap is set and
// already reached; the caller must answer 429 without touching a model.
inline bool try_admit(counters & c, int64_t max_inflight) {
    int64_t cur = c.inflight.load(std::memory_order_relaxed);
    for (;;) {
        if (max_inflight > 0 && cur >= max_inflight) {
            c.rejected_total.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (c.inflight.compare_exchange_weak(cur, cur + 1, std::memory_order_relaxed)) break;
    }
    c.admitted_total.fetch_add(1, std::memory_order_relaxed);
    return true;
}

inline void release(counters & c, int64_t duration_us) {
    c.inflight.fetch_sub(1, std::memory_order_relaxed);
    c.completed_total.fetch_add(1, std::memory_order_relaxed);
    c.duration_us_total.fetch_add(duration_us < 0 ? 0 : duration_us, std::memory_order_relaxed);
}

// Record usage exactly as the response reported it. Negative = not observed.
inline void record_usage(counters & c, int64_t prompt_tokens, int64_t completion_tokens) {
    if (prompt_tokens > 0) c.prompt_tokens_total.fetch_add(prompt_tokens, std::memory_order_relaxed);
    if (completion_tokens > 0) c.completion_tokens_total.fetch_add(completion_tokens, std::memory_order_relaxed);
}

inline std::string rejection_body(int64_t max_inflight) {
    std::ostringstream js;
    js << "{\"error\": {\"message\": \"server at capacity (" << max_inflight
       << " requests in flight)\", \"type\": \"rate_limit_error\", \"code\": \"server_at_capacity\"}}";
    return js.str();
}

// Prometheus text exposition (version 0.0.4).
inline std::string prometheus_text(const counters & c, const config & cfg) {
    const int64_t inflight = c.inflight.load(std::memory_order_relaxed);
    const int64_t running = c.running.load(std::memory_order_relaxed);
    const int64_t waiting = inflight > running ? inflight - running : 0;
    std::ostringstream o;
    auto metric = [&](const char * name, const char * type, const char * help, int64_t v) {
        o << "# HELP " << name << ' ' << help << "\n# TYPE " << name << ' ' << type << '\n' << name << ' ' << v << '\n';
    };
    metric("crispembed_requests_running", "gauge", "Requests holding a model lock.", running);
    metric("crispembed_requests_waiting", "gauge", "Admitted requests not holding a model lock.", waiting);
    metric("crispembed_requests_inflight", "gauge", "Admitted POST requests not yet finished.", inflight);
    metric("crispembed_requests_max_inflight", "gauge", "Admission cap (0 = uncapped).", cfg.max_inflight);
    metric("crispembed_requests_admitted_total", "counter", "Admitted POST requests.",
           c.admitted_total.load(std::memory_order_relaxed));
    metric("crispembed_requests_rejected_total", "counter", "POST requests rejected with 429 at admission.",
           c.rejected_total.load(std::memory_order_relaxed));
    metric("crispembed_requests_completed_total", "counter", "Admitted POST requests finished.",
           c.completed_total.load(std::memory_order_relaxed));
    {
        const int64_t us = c.duration_us_total.load(std::memory_order_relaxed);
        o << "# HELP crispembed_request_duration_seconds_sum Wall time of finished admitted requests.\n"
          << "# TYPE crispembed_request_duration_seconds_sum counter\n"
          << "crispembed_request_duration_seconds_sum " << us / 1000000 << '.';
        const int64_t frac = us % 1000000;
        o << (frac < 100000 ? "0" : "") << (frac < 10000 ? "0" : "") << (frac < 1000 ? "0" : "")
          << (frac < 100 ? "0" : "") << (frac < 10 ? "0" : "") << frac << '\n';
    }
    metric("crispembed_prompt_tokens_total", "counter", "Prompt tokens reported in API usage.",
           c.prompt_tokens_total.load(std::memory_order_relaxed));
    metric("crispembed_completion_tokens_total", "counter", "Completion tokens reported in API usage.",
           c.completion_tokens_total.load(std::memory_order_relaxed));
    return o.str();
}

} // namespace server_load
