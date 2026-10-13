#include "gemma_decision.h"
#include "crispembed_diff.h"
#include "core/clean_exit.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: model.gguf reference.gguf native-dump.gguf\n");
        return 2;
    }
#ifdef _WIN32
    _putenv_s("CRISPEMBED_GEMMA_DECISION", "1");
    _putenv_s("CRISPEMBED_GEMMA_DECISION_DUMP", "1");
    _putenv_s("CRISPEMBED_GEMMA_DECISION_DUMP_PATH", argv[3]);
#else
    setenv("CRISPEMBED_GEMMA_DECISION", "1", 1);
    setenv("CRISPEMBED_GEMMA_DECISION_DUMP", "1", 1);
    setenv("CRISPEMBED_GEMMA_DECISION_DUMP_PATH", argv[3], 1);
#endif
    crispembed_diff::Ref ref, native;
    if (!ref.load(argv[2])) return 1;
    std::unique_ptr<gemma_decision_context, decltype(&gemma_decision_free)> ctx(gemma_decision_init(argv[1], 2),
                                                                                gemma_decision_free);
    if (!ctx) return 1;
    std::vector<float> embedding;
    std::vector<int32_t> ids;
    if (!gemma_decision_encode(ctx.get(), ref.meta("ref.text"), embedding, &ids)) {
        fprintf(stderr, "%s\n", gemma_decision_error(ctx.get()));
        return 1;
    }
    auto expected_ids = ref.get_f32("input_ids");
    if (expected_ids.second != ids.size()) {
        fprintf(stderr, "token count mismatch %zu vs %zu\n", ids.size(), expected_ids.second);
        return 1;
    }
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] != (int32_t)expected_ids.first[i]) {
            fprintf(stderr, "token %zu mismatch %d vs %.0f\n", i, ids[i], expected_ids.first[i]);
            return 1;
        }
    if (!native.load(argv[3])) return 1;
    int pass = 0, fail = 0;
    std::vector<std::string> names{ "post_embed", "ple" };
    for (int i = 0; i < 24; ++i) names.push_back("layer_" + std::to_string(i));
    names.push_back("final_norm");
    names.push_back("tokens");
    names.push_back("embedding");
    for (auto & name : names) {
        auto data = name == "embedding" ? std::make_pair(embedding.data(), embedding.size()) : native.get_f32(name);
        auto r = ref.compare(name, data.first, data.second);
        auto expected = ref.get_f32(name);
        double difference = 0;
        if (expected.second == data.second)
            for (size_t i = 0; i < data.second; ++i)
                difference += std::pow(double(data.first[i]) - expected.first[i], 2);
        else
            difference = INFINITY;
        double relative = std::sqrt(difference) / std::max(double(r.ref_norm), 1e-12);
        const bool ok = r.found && r.cos_min >= 0.99999f && relative <= 0.003;
        printf("%s %-16s cos_min=%.9f cos_global=%.9f |mine|=%.6f |ref|=%.6f max_abs=%.7f rel=%.8f\n",
               ok ? "PASS" : "FAIL", name.c_str(), r.cos_min, r.cos_global, r.mine_norm, r.ref_norm, r.max_abs,
               relative);
        ok ? ++pass : ++fail;
    }
    printf("layer checks: %d passed / %d failed; exact IDs: %zu\n", pass, fail, ids.size());
    return fail ? 1 : 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
