// test_low_memory.cpp — hermetic guard for core/low_memory.h, the automatic
// skip of optional F32 weight copies (MiniCPM-OCR, Unlimited-OCR-2). No
// weights, no GGUF, no network.
#include "core/low_memory.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <cstdlib>

static int g_checks = 0, g_failures = 0;

static void expect(bool ok, const char * what) {
    g_checks++;
    if (!ok) {
        g_failures++;
        std::fprintf(stderr, "FAIL %s\n", what);
    }
}

static void set_var(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

static int crispembed_test_main() {
    using namespace core_lowmem;
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;

    expect(parse_mode(nullptr) == mode::automatic, "unset is automatic");
    expect(parse_mode("") == mode::automatic, "empty is automatic");
    expect(parse_mode("0") == mode::disabled, "0 disables the check");
    expect(parse_mode("1") == mode::forced_on, "1 forces the low-memory path");
    expect(parse_mode("yes") == mode::forced_on, "other non-empty values force it");

    // Forced: always skip, whatever the memory.
    expect(decide(mode::forced_on, 3 * GiB, int64_t(64 * GiB)) == decision::skip_forced, "forced, plenty");
    expect(decide(mode::forced_on, 3 * GiB, -1) == decision::skip_forced, "forced, unknown");
    // Disabled: never skip.
    expect(decide(mode::disabled, 3 * GiB, int64_t(1 * GiB)) == decision::keep_copies, "disabled, low");
    // Automatic.
    expect(decide(mode::automatic, 3 * GiB, -1) == decision::keep_copies, "unknown memory keeps copies");
    expect(decide(mode::automatic, 0, 0) == decision::keep_copies, "no copies, nothing to skip");
    expect(decide(mode::automatic, 3 * GiB, int64_t(64 * GiB)) == decision::keep_copies, "plenty keeps copies");
    expect(decide(mode::automatic, 3 * GiB, int64_t(4 * GiB)) == decision::keep_copies,
           "exactly extra + margin keeps copies");
    expect(decide(mode::automatic, 3 * GiB, int64_t(4 * GiB) - 1) == decision::skip_low_memory,
           "one byte short skips");
    expect(decide(mode::automatic, 3 * GiB, int64_t(2 * GiB)) == decision::skip_low_memory, "low memory skips");
    expect(decide(mode::automatic, 3 * GiB, int64_t(3 * GiB), 0) == decision::keep_copies, "zero margin");

    // Measurement hook and the env wrapper.
    const char * env = "CRISPEMBED_TEST_LOW_MEMORY";
    set_var("CRISPEMBED_RAM_GUARD_AVAILABLE_MB", "2048");
    expect(available_physical_bytes() == int64_t(2048) * 1024 * 1024, "available-memory test hook");
    set_var(env, nullptr);
    expect(skip_f32_copies("test", env, 3 * GiB), "wrapper: automatic, 2 GiB < 4 GiB skips");
    expect(!skip_f32_copies("test", env, 512ull * 1024 * 1024), "wrapper: automatic, small copies kept");
    set_var(env, "0");
    expect(!skip_f32_copies("test", env, 3 * GiB), "wrapper: =0 keeps copies");
    set_var(env, "1");
    expect(skip_f32_copies("test", env, 1), "wrapper: =1 skips");
    set_var(env, nullptr);
    set_var("CRISPEMBED_RAM_GUARD_AVAILABLE_MB", nullptr);
    // Real measurement: either unknown (-1) or a positive byte count.
    const int64_t real = available_physical_bytes();
    expect(real == -1 || real > 0, "real measurement is -1 or positive");

    std::printf("test_low_memory: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
