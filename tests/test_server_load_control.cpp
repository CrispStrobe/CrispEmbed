// Host-only unit test for crispembed-server's load control and OpenAI-shaped
// OCR request parsing (examples/server/load_control.h, openai_compat.h).
//
// No model, no ggml, no httplib. Covers: admission cap and counters, the
// thread-local "running" depth (a request holding two model locks counts once),
// env knob parsing (0 / junk = off), the Prometheus exposition, strict base64 / data-URL decoding with a size cap,
// and the chat request contract (one image, last user message, explicit
// rejections instead of silent drops).

#include "core/clean_exit.h"
#include "load_control.h"
#include "openai_compat.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

static int g_failures = 0;

static void check(const char * name, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) g_failures++;
}

static bool contains(const std::string & s, const std::string & sub) {
    return s.find(sub) != std::string::npos;
}

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v)
        setenv(k, v, 1);
    else
        unsetenv(k);
#endif
}

static void test_admission() {
    std::printf("admission\n");
    server_load::counters c;
    check("uncapped admits", server_load::try_admit(c, 0) && server_load::try_admit(c, 0));
    check("inflight counts admitted", c.inflight.load() == 2);
    server_load::release(c, 10);
    server_load::release(c, 20);
    check("release returns to zero", c.inflight.load() == 0 && c.completed_total.load() == 2);
    check("duration accumulates", c.duration_us_total.load() == 30);

    check("cap 2: first", server_load::try_admit(c, 2));
    check("cap 2: second", server_load::try_admit(c, 2));
    check("cap 2: third rejected", !server_load::try_admit(c, 2));
    check("rejection counted, not admitted", c.rejected_total.load() == 1 && c.inflight.load() == 2);
    server_load::release(c, 0);
    check("cap frees a slot after release", server_load::try_admit(c, 2));
    check("admitted_total", c.admitted_total.load() == 5);

    // Concurrent: never more than the cap in flight.
    server_load::counters k;
    std::atomic<int64_t> peak{ 0 };
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; t++) {
        ts.emplace_back([&] {
            for (int i = 0; i < 2000; i++) {
                if (!server_load::try_admit(k, 3)) continue;
                const int64_t now = k.inflight.load();
                int64_t p = peak.load();
                while (now > p && !peak.compare_exchange_weak(p, now)) {
                }
                server_load::release(k, 0);
            }
        });
    }
    for (auto & t : ts) t.join();
    check("concurrent peak <= cap", peak.load() <= 3);
    check("concurrent ends at zero", k.inflight.load() == 0);
    check("admitted + rejected = attempts", k.admitted_total.load() + k.rejected_total.load() == 16000);

    const std::string body = server_load::rejection_body(4);
    check("429 body names the cap", contains(body, "4 requests in flight") && contains(body, "server_at_capacity"));
}

static void test_counted_mutex() {
    std::printf("counted_mutex\n");
    server_load::counters c;
    server_load::counted_mutex a(&c), b(&c);
    {
        std::lock_guard<server_load::counted_mutex> la(a);
        check("one lock -> running 1", c.running.load() == 1);
        {
            std::lock_guard<server_load::counted_mutex> lb(b);
            check("nested lock still running 1", c.running.load() == 1);
        }
        check("inner unlock keeps running 1", c.running.load() == 1);
    }
    check("outer unlock -> running 0", c.running.load() == 0);
    check("try_lock counts", a.try_lock() && c.running.load() == 1);
    a.unlock();
    check("try_lock unlock -> 0", c.running.load() == 0);

    // Two threads each holding a different lock: running 2.
    std::atomic<int> ready{ 0 };
    std::atomic<bool> go{ false };
    auto hold = [&](server_load::counted_mutex & m) {
        std::lock_guard<server_load::counted_mutex> l(m);
        ready++;
        while (!go.load()) std::this_thread::yield();
    };
    std::thread t1(hold, std::ref(a)), t2(hold, std::ref(b));
    while (ready.load() < 2) std::this_thread::yield();
    check("two threads -> running 2", c.running.load() == 2);
    go = true;
    t1.join();
    t2.join();
    check("both released -> 0", c.running.load() == 0);
}

static void test_env() {
    std::printf("env knobs\n");
    set_env("CRISPEMBED_SERVER_METRICS", nullptr);
    set_env("CRISPEMBED_SERVER_MAX_INFLIGHT", nullptr);
    set_env("CRISPEMBED_SERVER_RETRY_AFTER", nullptr);
    auto d = server_load::config::from_env();
    check("defaults: metrics off, uncapped, no Retry-After", !d.metrics && d.max_inflight == 0 && d.retry_after < 0);

    set_env("CRISPEMBED_SERVER_METRICS", "0");
    set_env("CRISPEMBED_SERVER_MAX_INFLIGHT", "0");
    auto z = server_load::config::from_env();
    check("=0 means off", !z.metrics && z.max_inflight == 0);

    set_env("CRISPEMBED_SERVER_METRICS", "1");
    set_env("CRISPEMBED_SERVER_MAX_INFLIGHT", "8");
    set_env("CRISPEMBED_SERVER_RETRY_AFTER", "2");
    auto on = server_load::config::from_env();
    check("set values parse", on.metrics && on.max_inflight == 8 && on.retry_after == 2);

    for (const char * junk : { "-3", "8x", "", "abc", "99999999999999999999999" }) {
        set_env("CRISPEMBED_SERVER_MAX_INFLIGHT", junk);
        check((std::string("junk cap '") + junk + "' -> uncapped").c_str(),
              server_load::config::from_env().max_inflight == 0);
    }
    set_env("CRISPEMBED_SERVER_METRICS", nullptr);
    set_env("CRISPEMBED_SERVER_MAX_INFLIGHT", nullptr);
    set_env("CRISPEMBED_SERVER_RETRY_AFTER", nullptr);
}

static void test_prometheus() {
    std::printf("prometheus\n");
    server_load::counters c;
    server_load::config cfg;
    cfg.max_inflight = 6;
    server_load::try_admit(c, 0);
    server_load::try_admit(c, 0);
    server_load::try_admit(c, 0);
    c.running = 1;
    c.duration_us_total = 1234567;
    server_load::record_usage(c, 40, -1);
    server_load::record_usage(c, -1, 7);
    const std::string t = server_load::prometheus_text(c, cfg);
    check("running", contains(t, "\ncrispembed_requests_running 1\n"));
    check("waiting = inflight - running", contains(t, "\ncrispembed_requests_waiting 2\n"));
    check("cap", contains(t, "\ncrispembed_requests_max_inflight 6\n"));
    check("duration seconds", contains(t, "\ncrispembed_request_duration_seconds_sum 1.234567\n"));
    check("tokens", contains(t, "\ncrispembed_prompt_tokens_total 40\n") &&
                        contains(t, "\ncrispembed_completion_tokens_total 7\n"));
    check("no foreign-dialect series", !contains(t, "llamacpp"));
    check("TYPE lines", contains(t, "# TYPE crispembed_requests_admitted_total counter\n"));
    c.duration_us_total = 5;
    check("sub-second padding",
          contains(server_load::prometheus_text(c, cfg), "crispembed_request_duration_seconds_sum 0.000005\n"));
    c.running = 9; // the two counters are read separately, so a snapshot can see running > inflight: clamp
    check("waiting never negative",
          contains(server_load::prometheus_text(c, cfg), "\ncrispembed_requests_waiting 0\n"));
}

static void test_base64() {
    std::printf("base64 / data URL\n");
    std::string out;
    check("plain", server_compat::base64_decode("aGVsbG8=", 0, out) && out == "hello");
    check("no padding", server_compat::base64_decode("aGVsbG8", 0, out) && out == "hello");
    check("two pad", server_compat::base64_decode("aGk=", 0, out) && out == "hi");
    check("wrapped", server_compat::base64_decode("aGVs\nbG8=\n", 0, out) && out == "hello");
    check("binary", server_compat::base64_decode("AP8=", 0, out) && out.size() == 2 && (unsigned char)out[0] == 0 &&
                        (unsigned char)out[1] == 255);
    check("bad char rejected", !server_compat::base64_decode("aGV*bG8=", 0, out));
    check("data after pad rejected", !server_compat::base64_decode("aGk=aGk=", 0, out));
    check("lone sextet rejected", !server_compat::base64_decode("aGVsb", 0, out));
    check("wrong pad rejected", !server_compat::base64_decode("aGk==", 0, out));
    check("limit enforced", !server_compat::base64_decode("aGVsbG8=", 0, out, 3));

    std::string bytes, err;
    check("data URL", server_compat::decode_data_url("data:image/png;base64,aGVsbG8=", bytes, err) && bytes == "hello");
    check("data URL with params",
          server_compat::decode_data_url("data:image/png;name=a.png;base64,aGk=", bytes, err) && bytes == "hi");
    check("http rejected",
          !server_compat::decode_data_url("https://example.com/a.png", bytes, err) && contains(err, "not fetched"));
    check("file rejected", !server_compat::decode_data_url("file:///etc/passwd", bytes, err));
    check("non-base64 data URL rejected", !server_compat::decode_data_url("data:text/plain,hello", bytes, err));
    check("empty rejected", !server_compat::decode_data_url("data:image/png;base64,", bytes, err));
    const std::string big = "data:image/png;base64," + std::string(4000, 'A');
    check("oversize rejected before decode",
          !server_compat::decode_data_url(big, bytes, err, 100) && contains(err, "size limit"));
}

static const char * IMG = "data:image/png;base64,aGk=";

static std::string chat(const std::string & content, const std::string & extra = "") {
    return std::string("{\"model\": \"ovisocr2\", ") + extra +
           "\"messages\": [{\"role\": \"user\", \"content\": " + content + "}]}";
}

static void test_chat() {
    std::printf("chat request\n");
    server_compat::chat_ocr_request r;
    std::string err;
    const std::string parts = std::string("[{\"type\": \"text\", \"text\": \"Read the page.\"}, "
                                          "{\"type\": \"image_url\", \"image_url\": {\"url\": \"") +
                              IMG + "\", \"detail\": \"high\"}}]";
    check("standard request", server_compat::parse_chat_ocr_request(chat(parts, "\"max_tokens\": 512, "), r, err) &&
                                  r.prompt == "Read the page." && r.image_url == IMG && r.max_tokens == 512 &&
                                  r.model == "ovisocr2");
    check("max_completion_tokens preferred",
          server_compat::parse_chat_ocr_request(chat(parts, "\"max_completion_tokens\": 64, \"max_tokens\": 512, "), r,
                                                err) &&
              r.max_tokens == 64);
    check("image only, no prompt",
          server_compat::parse_chat_ocr_request(
              chat(std::string("[{\"type\": \"image_url\", \"image_url\": {\"url\": \"") + IMG + "\"}}]"), r, err) &&
              r.prompt.empty() && r.max_tokens == 0);
    check("legacy flat image_url string",
          server_compat::parse_chat_ocr_request(
              chat(std::string("[{\"type\": \"image_url\", \"image_url\": \"") + IMG + "\"}]"), r, err) &&
              r.image_url == IMG);
    check("text parts joined", server_compat::parse_chat_ocr_request(
                                   chat(std::string("[{\"type\": \"text\", \"text\": \"a\"}, {\"type\": \"text\", "
                                                    "\"text\": \"b\"}, {\"type\": \"image_url\", \"image_url\": "
                                                    "{\"url\": \"") +
                                        IMG + "\"}}]"),
                                   r, err) &&
                                   r.prompt == "a\nb");
    const std::string sys_then_user =
        std::string("{\"messages\": [{\"role\": \"system\", \"content\": \"be terse\"}, {\"role\": \"user\", "
                    "\"content\": [{\"type\": \"text\", \"text\": \"OCR\"}, {\"type\": \"image_url\", \"image_url\": "
                    "{\"url\": \"") +
        IMG + "\"}}]}]}";
    check("system message ignored, last user used",
          server_compat::parse_chat_ocr_request(sys_then_user, r, err) && r.prompt == "OCR");
    check("escaped prompt decoded",
          server_compat::parse_chat_ocr_request(
              chat(std::string("[{\"type\": \"text\", \"text\": \"say \\\"hi\\\"\\n\"}, {\"type\": \"image_url\", "
                               "\"image_url\": {\"url\": \"") +
                   IMG + "\"}}]"),
              r, err) &&
              r.prompt == "say \"hi\"\n");

    check("string content rejected (no image)",
          !server_compat::parse_chat_ocr_request(chat("\"hello\""), r, err) && contains(err, "no image"));
    check("two images rejected",
          !server_compat::parse_chat_ocr_request(
              chat(std::string("[{\"type\": \"image_url\", \"image_url\": {\"url\": \"") + IMG +
                   "\"}}, {\"type\": \"image_url\", \"image_url\": {\"url\": \"" + IMG + "\"}}]"),
              r, err) &&
              contains(err, "exactly one"));
    check("stream rejected",
          !server_compat::parse_chat_ocr_request(chat(parts, "\"stream\": true, "), r, err) && contains(err, "stream"));
    check("stream false ok", server_compat::parse_chat_ocr_request(chat(parts, "\"stream\": false, "), r, err));
    check("n=2 rejected", !server_compat::parse_chat_ocr_request(chat(parts, "\"n\": 2, "), r, err));
    check("n=1 ok", server_compat::parse_chat_ocr_request(chat(parts, "\"n\": 1, "), r, err));
    check("tools rejected",
          !server_compat::parse_chat_ocr_request(chat(parts, "\"tools\": [{\"type\": \"function\"}], "), r, err));
    check("bad max_tokens rejected",
          !server_compat::parse_chat_ocr_request(chat(parts, "\"max_tokens\": -1, "), r, err));
    check("fractional max_tokens rejected",
          !server_compat::parse_chat_ocr_request(chat(parts, "\"max_tokens\": 1.5, "), r, err));
    check("unknown part type rejected",
          !server_compat::parse_chat_ocr_request(chat("[{\"type\": \"input_audio\", \"input_audio\": {}}]"), r, err) &&
              contains(err, "input_audio"));
    check("no messages rejected", !server_compat::parse_chat_ocr_request("{\"model\": \"x\"}", r, err));
    check("empty messages rejected", !server_compat::parse_chat_ocr_request("{\"messages\": []}", r, err));
    check("no user rejected", !server_compat::parse_chat_ocr_request(
                                  "{\"messages\": [{\"role\": \"system\", \"content\": \"x\"}]}", r, err));
    check("malformed rejected", !server_compat::parse_chat_ocr_request("{\"messages\": [{\"role\": ", r, err));
    // A decoy nested "messages" key must not be read as the top-level field.
    check("nested decoy ignored",
          !server_compat::parse_chat_ocr_request(
              std::string("{\"meta\": {\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", "
                          "\"image_url\": {\"url\": \"") +
                  IMG + "\"}}]}]}}",
              r, err));
}

static int crispembed_test_main() {
    test_admission();
    test_counted_mutex();
    test_env();
    test_prometheus();
    test_base64();
    test_chat();
    std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}

// Route through core_util::clean_exit per the tools/check_test_clean_exit.sh guard.
int main() {
    core_util::clean_exit(crispembed_test_main());
}
