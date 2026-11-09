// SPDX-License-Identifier: MIT
//
// Model-free, network-free contract for per-call licence acceptance:
//
//   1. crispembed_resolve_model_accepting accepts the exact SPDX tag
//   2. ... refuses a model whose licence is a different tag
//   3. ... refuses the blanket values "all" / "*" (and empty / NULL)
//   4. crispembed_resolve_model keeps its behaviour (exact or blanket accept,
//      refusal without acceptance)
//   5. CRISPEMBED_ACCEPT_LICENSE keeps its behaviour for both APIs
//
// Every refusal happens before any download, and the cache directory is an
// empty temporary one, so nothing here touches the network. The "accepted"
// direction is checked on the shared decision rule (core/license_gate.h),
// because the C API would go on to download.
#include "crispembed.h"
#include "core/clean_exit.h"
#include "core/license_gate.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace {
int checks = 0;
void check(bool ok, const char * label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        core_util::clean_exit(1);
    }
}
void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
bool empty(const char * s) {
    return !s || !*s;
}
} // namespace

static int crispembed_test_main(int, char **) {
#ifndef _WIN32
    // The legacy API prompts on a TTY; the contract under test is the
    // non-interactive one an app sees.
    if (!std::freopen("/dev/null", "r", stdin)) return 2;
#endif
    const auto cache = std::filesystem::temp_directory_path() / "crispembed-license-consent-test";
    std::filesystem::remove_all(cache);
    std::filesystem::create_directories(cache);
    set_env("CRISPEMBED_CACHE_DIR", cache.string().c_str());
    set_env("CRISPEMBED_ACCEPT_LICENSE", nullptr);
    set_env("CRISPEMBED_OFFLINE", nullptr);
    set_env("HF_HUB_OFFLINE", nullptr);
    crispembed_set_offline(0);

    // Find a restricted and a permissive registry entry through the new getter.
    std::string restricted, restricted_spdx, permissive;
    for (int i = 0; i < crispembed_n_models(); i++) {
        const char * reg = crispembed_model_name(i);
        const char *name = nullptr, *spdx = nullptr, *card = nullptr, *display = nullptr;
        const int rc = crispembed_model_license_info(reg, &name, &spdx, &card, &display);
        check(rc >= 0, "every registry name has licence info");
        if (std::strcmp(name, reg) != 0) continue; // substring alias; skip
        check(!empty(display), "display name present");
        if (rc == 1 && restricted.empty() && !empty(spdx) && std::strncmp(spdx, "cc-by-nc", 8) == 0) {
            restricted = name;
            restricted_spdx = spdx;
            check(!empty(card), "restricted entry carries a model card URL");
        }
        if (rc == 0 && permissive.empty()) permissive = name;
    }
    check(!restricted.empty(), "registry has a cc-by-nc entry");
    check(!permissive.empty(), "registry has a permissive entry");
    {
        const char * spdx = "x";
        check(crispembed_model_license_info("no-such-model-zzz", nullptr, &spdx, nullptr, nullptr) == -1,
              "unknown model reports -1");
        check(spdx == nullptr, "unknown model clears out pointers");
    }

    const char * r = restricted.c_str();
    const char * s = restricted_spdx.c_str();

    // 1. Exact tag accepted, per call, blanket not needed.
    check(core_license::accepted(s, restricted_spdx, false, nullptr), "1: exact tag accepted by per-call rule");

    // 2. A different tag is refused.
    check(!core_license::accepted(s, "gemma", false, nullptr), "2: different tag refused by rule");
    check(empty(crispembed_resolve_model_accepting(r, 1, "gemma")), "2: different tag refused by C API");
    check(empty(crispembed_resolve_model_accepting(r, 1, "cc-by-nc-sa-4.0-not-this")),
          "2: near-miss tag refused by C API");

    // 3. Blanket and empty values are refused by the new API.
    check(!core_license::accepted(s, "all", false, nullptr), "3: all refused by per-call rule");
    check(!core_license::accepted(s, "*", false, nullptr), "3: * refused by per-call rule");
    check(empty(crispembed_resolve_model_accepting(r, 1, "all")), "3: all refused by C API");
    check(empty(crispembed_resolve_model_accepting(r, 1, "*")), "3: * refused by C API");
    check(empty(crispembed_resolve_model_accepting(r, 1, "")), "3: empty refused by C API");
    check(empty(crispembed_resolve_model_accepting(r, 1, nullptr)), "3: NULL refused by C API");
    check(empty(crispembed_resolve_model_accepting(permissive.c_str(), 1, "all")),
          "3: all refused even for a permissive model");

    // 4. Legacy API: blanket and exact still accepted by its rule, and with no
    // acceptance a restricted download is still refused despite auto_download.
    check(core_license::accepted(s, "all", true, nullptr), "4: legacy rule accepts all");
    check(core_license::accepted(s, "*", true, nullptr), "4: legacy rule accepts *");
    check(core_license::accepted(s, restricted_spdx, true, nullptr), "4: legacy rule accepts exact tag");
    check(!core_license::accepted(s, "", true, nullptr), "4: legacy rule refuses nothing accepted");
    check(empty(crispembed_resolve_model(r, 1)), "4: legacy API refuses restricted without acceptance");

    // 5. CRISPEMBED_ACCEPT_LICENSE: unchanged meaning, for both APIs.
    check(core_license::accepted(s, "", true, s), "5: env exact tag accepted");
    check(core_license::accepted(s, "", true, "all"), "5: env all accepted (legacy semantics)");
    check(core_license::accepted(s, "gemma", false, "all"), "5: env all still applies to per-call API");
    check(!core_license::accepted(s, "", true, "gemma"), "5: env different tag refused");
    set_env("CRISPEMBED_ACCEPT_LICENSE", "gemma");
    check(empty(crispembed_resolve_model(r, 1)), "5: legacy API refuses with mismatched env");
    check(empty(crispembed_resolve_model_accepting(r, 1, "lfm1.0")), "5: per-call refuses with mismatched env");
    check(empty(crispembed_resolve_model_accepting(r, 1, "all")), "5: env does not unlock all in per-call");
    set_env("CRISPEMBED_ACCEPT_LICENSE", nullptr);

    std::filesystem::remove_all(cache);
    std::printf("license consent: %d checks passed\n", checks);
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
