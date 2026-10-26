// Actual native tokenizer controls, no model inference/weights allocation.
#include "core/clean_exit.h"
#include "ultrabert_profile.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace {
void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_ULTRABERT_TRUNCATE", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_ULTRABERT_TRUNCATE", value, 1);
    else
        unsetenv("CRISPEMBED_ULTRABERT_TRUNCATE");
#endif
}
int run(int argc, char ** argv) {
    if (!core_env::on("CRISPEMBED_ULTRABERT")) return 2;
    ultrabert_profile::require(argc == 2, "Expected qualified F32 GGUF");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(core_gguf::open_metadata(argv[1]), gguf_free);
    ultrabert_profile::validate(meta.get());
    BPETokenizer raw;
    std::vector<std::string> vocab;
    ultrabert_profile::require(ultrabert_profile::load_tokenizer(meta.get(), raw, vocab), "Tokenizer load failed");
    std::string long_text;
    for (int i = 0; i < 600; ++i) long_text += " evidence";
    const std::string query = std::string(ultrabert_profile::query_prefix()) + long_text;
    std::vector<int32_t> ids;
    for (const auto & text : { long_text, query }) {
        const auto untruncated = raw.encode(text).ids;
        ultrabert_profile::require(untruncated.size() >= 512, "Control is not long enough");
        for (const char * value : { static_cast<const char *>(nullptr), "0" }) {
            gate(value);
            ids = { 4, 5 };
            ultrabert_profile::require(!ultrabert_profile::tokenize(raw, text, ids) && ids.empty(),
                                       "Unset/zero truncation accepted overflow or retained IDs");
            ultrabert_profile::require(ultrabert_profile::tokenize(raw, "recovery", ids) && ids[0] == 0,
                                       "Failure recovery failed");
        }
        gate("1");
        ultrabert_profile::require(ultrabert_profile::tokenize(raw, text, ids) && ids.size() == 512 && ids[0] == 0 &&
                                       std::equal(ids.begin() + 1, ids.end(), untruncated.begin()),
                                   "Truncation did not retain BOS plus exact first511 text/prefix IDs");
        const auto first = ids;
        ultrabert_profile::require(ultrabert_profile::tokenize(raw, text, ids) && ids == first,
                                   "Repeated truncation changed IDs");
        ultrabert_profile::require(!ultrabert_profile::tokenize(raw, "<|mask|>", ids) && ids.empty(),
                                   "Truncation admitted reserved token surface");
        ultrabert_profile::require(!ultrabert_profile::tokenize(raw, std::string(65537, 'x'), ids) && ids.empty(),
                                   "Truncation admitted byte overflow");
    }
    gate(nullptr);
    printf(
        "PASS actual UltraBERT tokenizer unset/zero rejection, recovery, BOS/prefix/right512, repeat and exclusions\n");
    return 0;
}
} // namespace
int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        core_util::clean_exit(1);
    }
}
