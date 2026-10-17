// Test-only native BPE + declared UltraBERT template; no inference or public dispatch.
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "tokenizer.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
int run(int argc, char ** argv) {
    if (!core_env::on("CRISPEMBED_ULTRABERT")) return 2;
    require(argc == 2 || argc == 3, "Expected tokenizer fixture [--terminal-sep|--mask-zero]");
    const std::string mode = argc == 3 ? argv[2] : "";
    require(mode.empty() || mode == "--terminal-sep" || mode == "--mask-zero", "Unknown negative control");
    auto * meta = core_gguf::open_metadata(argv[1]);
    require(meta != nullptr, "Fixture load failed");
    std::vector<std::string> vocab, merges;
    core_gguf::load_vocab_array(meta, "tokenizer.ggml.tokens", vocab);
    core_gguf::load_vocab_array(meta, "tokenizer.ggml.merges", merges);
    require(vocab.size() == 49152 && merges.size() == 48900 && vocab[0] == "<|endoftext|>" && vocab[15] == "<|sep|>" &&
                core_gguf::kv_str(meta, "ref.revision", "") == "367d5343b5a0164d57b810cedf5de4b8983d5984",
            "Wrong pinned tokenizer fixture");
    BPETokenizer raw;
    // Raw pieces only: UltraBERT has BOS once, a separator only BETWEEN a pair,
    // and no terminal EOS. Do not use the generic BERT pair template.
    require(raw.load(vocab, merges, -1, 0, -1, -1, false, 1048576), "Native vocabulary load failed");
    raw.set_gpt2_regex_pretok(true);
    const auto rows = core_gguf::kv_u32(meta, "ref.rows", 0);
    require(rows == 8, "Incomplete original case rows");
    int failures = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        const std::string key = "row." + std::to_string(row) + ".";
        const std::string text = core_gguf::kv_str(meta, key + "text", "");
        const std::string pair = core_gguf::kv_str(meta, key + "pair", "");
        const auto expected = core_gguf::kv_i32_array(meta, key + "ids");
        const auto expected_mask = core_gguf::kv_i32_array(meta, key + "mask");
        auto encoded = raw.encode(text);
        std::vector<int32_t> ids = { 0 };
        ids.insert(ids.end(), encoded.ids.begin(), encoded.ids.end());
        if (core_gguf::kv_bool(meta, key + "has_pair", false)) {
            ids.push_back(15);
            encoded = raw.encode(pair);
            ids.insert(ids.end(), encoded.ids.begin(), encoded.ids.end());
        }
        if (mode == "--terminal-sep") ids.push_back(15);
        require(ids.size() <= 512 && !expected.empty() && expected.size() <= 512 &&
                    expected.size() == expected_mask.size(),
                "Unsupported bounded case");
        std::vector<int32_t> mask(ids.size(), 1);
        if (ids.size() < expected.size()) {
            ids.resize(expected.size(), 0);
            mask.resize(expected.size(), 0);
        }
        if (mode == "--mask-zero") {
            for (size_t i = 0; i < ids.size(); ++i) mask[i] = ids[i] != 0;
        }
        printf("IDS\t%u", row);
        for (int32_t value : ids) printf("\t%d", value);
        printf("\nMASK\t%u", row);
        for (int32_t value : mask) printf("\t%d", value);
        const bool pass = ids == expected && mask == expected_mask;
        printf("\nROW\t%u\t%d\n", row, pass ? 1 : 0);
        failures += !pass;
    }
    core_gguf::free_metadata(meta);
    printf("TOTAL\t%u\t%d\n", rows, failures);
    return failures ? 1 : 0;
}
int checked_main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & error) {
        fprintf(stderr, "UltraBERT tokenizer audit: %s\n", error.what());
        return 2;
    }
}
} // namespace
int main(int argc, char ** argv) {
    core_util::clean_exit(checked_main(argc, argv));
}
