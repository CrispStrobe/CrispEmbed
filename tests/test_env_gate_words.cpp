#include "core/env_gate.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <cstdlib>

static int run() {
    struct Case {
        const char * value;
        bool on;
        bool off;
    };
    const Case cases[] = { { "0", false, true },  { "false", false, true }, { "off", false, true },
                           { "no", false, true }, { "FALSE", false, true }, { "", false, false },
                           { "1", true, false },  { "true", true, false },  { " off", true, false } };
    for (const auto & c : cases) {
#ifdef _WIN32
        _putenv_s("CRISPEMBED_TEST_ENV_WORD", c.value);
#else
        setenv("CRISPEMBED_TEST_ENV_WORD", c.value, 1);
#endif
        if (core_env::on("CRISPEMBED_TEST_ENV_WORD") != c.on ||
            core_env::explicitly_off("CRISPEMBED_TEST_ENV_WORD") != c.off) {
            std::fprintf(stderr, "unexpected gate result for '%s'\n", c.value);
            return 1;
        }
    }
    return 0;
}

int main() {
    core_util::clean_exit(run());
}
