// Read-only public registry metadata against the production model manager.
// No model files, native inference, cache directory or network are needed.
#include "crispembed.h"
#include "core/clean_exit.h"
#include "model_mgr.h"
#include "model_hashes.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>

#ifdef NDEBUG
#error "Metadata contract checks require assertions"
#endif

static void assert_empty(const crispembed_model_download_info_v1 & info) {
    assert(!info.name && !info.license && !info.model_card_url);
    assert(!info.requires_license_acceptance && !info.has_companion);
    assert(!info.primary.filename && !info.primary.url && !info.primary.sha256);
    assert(!info.companion.filename && !info.companion.url && !info.companion.sha256);
}

static void check_pin(const crispembed_model_download_file_v1 & file) {
    assert(file.filename && *file.filename);
    // A registered local conversion/custom model has no downloadable URL.
    // Its metadata must stay truthful and must not acquire a fabricated pin.
    if (!file.url || !*file.url) {
        assert(!file.sha256);
        return;
    }
    assert(std::strncmp(file.url, "https://", 8) == 0);
    const char * expected = crispembed_mgr::model_pinned_sha256(file.url);
    if (!expected) {
        assert(!file.sha256);
        return;
    }
    assert(file.sha256 && std::strcmp(file.sha256, expected) == 0);
    assert(std::strlen(file.sha256) == 64);
    assert(std::strspn(file.sha256, "0123456789abcdef") == 64);
}

static int crispembed_test_main() {
    crispembed_model_download_info_v1 info = {};
    assert(crispembed_get_model_download_info_v1(nullptr, &info) == -1);
    assert_empty(info);
    assert(crispembed_get_model_download_info_v1("ppocrv6-tiny-det", nullptr) == -1);
    assert(crispembed_get_model_download_info_v1("ppocrv6-tiny-det", &info) == 1);
    assert(std::strcmp(info.primary.filename, "PP-OCRv6_tiny_det-f16.gguf") == 0);
    assert(std::strcmp(info.primary.sha256, "3b165a16a33b05521a6b374837dad9beff5e8e89164e700b9abe187f9923de50") == 0);
    assert(!info.requires_license_acceptance && !info.has_companion);
    const char * stable_name = info.name;
    const char * stable_pin = info.primary.sha256;

    // Unknown, filename, path, substring and different case never fuzzy-match.
    for (const char * name : { "", "not-a-registered-model", "PP-OCRv6_tiny_det-f16.gguf", "/tmp/ppocrv6-tiny-det",
                               "ppocrv6", "PPOCRV6-TINY-DET" }) {
        assert(crispembed_get_model_download_info_v1("ppocrv6-tiny-det", &info) == 1);
        assert(crispembed_get_model_download_info_v1(name, &info) == 0);
        assert_empty(info);
    }
    assert(std::strcmp(stable_name, "ppocrv6-tiny-det") == 0);
    assert(std::strcmp(stable_pin, "3b165a16a33b05521a6b374837dad9beff5e8e89164e700b9abe187f9923de50") == 0);

    // This existing registry entry requires local conversion. Lookup succeeds,
    // but an installer must reject its empty source URL and absent digest.
    assert(crispembed_get_model_download_info_v1("easyocr-english-g2", &info) == 1);
    assert(std::strcmp(info.primary.filename, "easyocr-english-g2-f16.gguf") == 0);
    assert(info.primary.url && !*info.primary.url && !info.primary.sha256);
    assert(!info.has_companion);
    check_pin(info.primary);

    // Offline state does not hide metadata or trigger a cache/network operation.
    crispembed_mgr::set_offline(true);
    int count = 0;
    int companion_count = 0;
    int local_only_count = 0;
    for (int index = 0; index < crispembed_mgr::n_models(); ++index) {
        const char * name = crispembed_mgr::model_name(index);
        assert(crispembed_get_model_download_info_v1(name, &info) == 1);
        assert(std::strcmp(info.name, name) == 0);
        assert(std::strcmp(info.primary.filename, crispembed_mgr::model_filename(index)) == 0);
        assert(info.requires_license_acceptance ==
               (crispembed_mgr::license_requires_acceptance(crispembed_mgr::model_license(index)) ? 1 : 0));
        check_pin(info.primary);
        if (!info.primary.url || !*info.primary.url) ++local_only_count;
        if (info.has_companion) {
            check_pin(info.companion);
            ++companion_count;
        }
        ++count;
    }
    assert(count > 0 && companion_count > 0 && local_only_count > 0);
    assert(crispembed_get_model_download_info_v1("lfm2-vl", &info) == 1);
    assert(info.requires_license_acceptance == 1 && info.has_companion == 1);
    assert(std::strcmp(info.companion.filename, "mmproj-LFM2.5-VL-3B-F16.gguf") == 0);
    assert(std::strcmp(info.companion.sha256, "6f912eb9fcef619b31c9afff7662a222da9ce6db89327eb465dbafd2fb6a2d7c") == 0);
    std::printf("{\"registry_entries\":%d,\"companions\":%d,\"local_only\":%d}\n", count, companion_count,
                local_only_count);
    return 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
