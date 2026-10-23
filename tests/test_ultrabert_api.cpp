// Public text API against immutable original pooled vectors. Full intermediate
// audit remains separate in test-ultrabert-diff with its unchanged failing gates.
#include "crispembed.h"
#include "crispembed_diff.h"
#include "core/clean_exit.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char * what) {
    if (!ok) throw std::runtime_error(what);
}
std::string text_file(const char * path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "missing text input");
    return std::string(std::istreambuf_iterator<char>(in), {});
}
int run(int argc, char ** argv) {
    require(argc >= 3, "model reference textfile... or model --reject");
    using Owner = std::unique_ptr<crispembed_context, decltype(&crispembed_free)>;
    Owner ctx(crispembed_init(argv[1], 2), crispembed_free);
    if (std::string(argv[2]) == "--reject") {
        require(!ctx, "unsupported/disabled model loaded");
        printf("PASS: rejected unsupported/disabled profile\n");
        return 0;
    }
    require(ctx != nullptr, "public model load failed");
    const auto * hp = crispembed_get_hparams(ctx.get());
    require(hp && hp->n_embd == 768 && hp->n_max_tokens == 512 && hp->n_vocab == 49152 && hp->n_output == 768,
            "wrong public hyperparameters");
    require(crispembed_tokenizer_kind(ctx.get()) == 3 &&
                std::string(crispembed_token_str(ctx.get(), 0)) == "<|endoftext|>" &&
                crispembed_token_str(ctx.get(), -1) == nullptr && !crispembed_has_masked_lm(ctx.get()),
            "wrong tokenizer/head contract");
    require(std::string(crispembed_get_prefix(ctx.get())).empty(), "implicit query prompt");
    const char * query = crispembed_ctx_query_prefix(ctx.get());
    require(query != nullptr, "query prefix missing");
    crispembed_diff::Ref ref;
    require(ref.load(argv[2]), "frozen reference load failed");
    const auto shape = ref.shape("input_ids");
    require(shape.size() == 2 && shape[1] == argc - 3 && shape[0] > 0 && shape[0] <= 512, "wrong batch coverage");
    const int T = int(shape[0]), B = int(shape[1]);
    const auto original_ids = ref.get_f32("input_ids"), mask = ref.get_f32("attention_mask");
    require(original_ids.second == size_t(T * B) && mask.second == original_ids.second, "bad original IDs/mask");
    std::vector<std::string> texts;
    for (int b = 0; b < B; ++b) texts.push_back(text_file(argv[b + 3]));
    std::vector<float> raw, normalized;
    std::vector<const char *> inputs;
    for (int b = 0; b < B; ++b) {
        inputs.push_back(texts[b].c_str());
        int dim = -1, n = -1;
        const auto * tokens = crispembed_encode_tokens_raw(ctx.get(), texts[b].c_str(), &n, &dim);
        require(tokens && n > 0 && dim == 768, "raw text API failed");
        const auto * ids = crispembed_last_token_ids(ctx.get());
        require(ids && ids[0] == 0, "missing attended BOS");
        int active = 0;
        for (int t = 0; t < T; ++t)
            if (mask.first[b * T + t] == 1) {
                require(active < n && ids[active] == original_ids.first[b * T + t], "actual text IDs differ");
                ++active;
            }
        require(active == n, "token count/truncation differs");
        raw.insert(raw.end(), tokens, tokens + 768);
        printf("IDS\t%d", b);
        for (int i = 0; i < n; ++i) printf("\t%d", ids[i]);
        printf("\n");
        const auto * out = crispembed_encode(ctx.get(), texts[b].c_str(), &dim);
        require(out && dim == 768, "normalized text API failed");
        const std::vector<float> first(out, out + dim);
        normalized.insert(normalized.end(), first.begin(), first.end());
        out = crispembed_encode(ctx.get(), texts[b].c_str(), &dim);
        require(out && std::equal(first.begin(), first.end(), out), "repeat changed output");
    }
    int failed = 0;
    for (const auto & stage : { std::make_pair("cls_raw", &raw), std::make_pair("cls_normalized", &normalized) }) {
        const auto r = ref.compare(stage.first, stage.second->data(), stage.second->size(), 768);
        bool finite = std::all_of(stage.second->begin(), stage.second->end(), [](float x) { return std::isfinite(x); });
        bool pass = finite && r.found && std::isfinite(r.cos_min) && std::isfinite(r.max_abs) && r.cos_min >= 0.99999 &&
                    r.max_abs <= 0.002;
        printf("POOLED\t%s\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\n", stage.first, r.cos_min, r.max_abs, r.mean_abs,
               r.rms, r.mine_norm, r.ref_norm, pass ? 1 : 0);
        failed += !pass;
    }
    int dim = -1;
    const auto * batch = crispembed_encode_batch(ctx.get(), inputs.data(), B, &dim);
    require(batch && dim == 768 && std::equal(normalized.begin(), normalized.end(), batch),
            "batch order/output differs");
    Owner other(crispembed_init(argv[1], 1), crispembed_free);
    require(other != nullptr, "second context failed");
    crispembed_set_prefix(other.get(), query);
    const auto * own_prefix = crispembed_get_prefix(other.get());
    require(std::string(own_prefix) == query && std::string(crispembed_get_prefix(ctx.get())).empty(),
            "context prefix leaked");
    const auto explicit_query = std::string(query) + texts[0];
    const auto * a = crispembed_encode(other.get(), texts[0].c_str(), &dim);
    require(a && dim == 768, "explicit query failed");
    std::vector<float> query_vector(a, a + dim);
    a = crispembed_encode(ctx.get(), explicit_query.c_str(), &dim);
    require(a && dim == 768, "manual query failed");
    double prefix_error = 0;
    for (int i = 0; i < dim; ++i) prefix_error = std::max(prefix_error, double(std::abs(a[i] - query_vector[i])));
    require(prefix_error <= 0.002, "prefix differs across contexts/threads");
    dim = 91;
    require(!crispembed_encode(ctx.get(), nullptr, &dim) && dim == 0, "null text retained dimensions");
    dim = 91;
    require(!crispembed_encode(nullptr, "text", &dim) && dim == 0, "null context retained dimensions");
    require(!crispembed_encode(ctx.get(), "<|mask|>", &dim) && dim == 0, "reserved surface accepted");
    require(!crispembed_encode(ctx.get(), std::string(65537, 'x').c_str(), &dim) && dim == 0, "byte overflow accepted");
    std::string long_text;
    for (int i = 0; i < 513; ++i) long_text += " x";
    require(!crispembed_encode(ctx.get(), long_text.c_str(), &dim) && dim == 0, "token overflow truncated");
    crispembed_set_dim(ctx.get(), 128);
    require(!crispembed_encode(ctx.get(), "dimension", &dim) && dim == 0, "unqualified truncation accepted");
    crispembed_set_dim(ctx.get(), 0);
    const char * bad[] = { "valid", nullptr };
    require(!crispembed_encode_batch(ctx.get(), bad, 2, &dim) && dim == 0, "partial batch returned");
    int n = 7;
    require(!crispembed_encode_tokens_raw(ctx.get(), "<|mask|>", &n, &dim) && n == 0 && dim == 0 &&
                crispembed_last_token_ids(ctx.get()) == nullptr,
            "failed token call retained IDs");
    require(crispembed_encode(ctx.get(), texts[0].c_str(), &dim) && dim == 768, "failure recovery failed");
    printf("TOTAL\t%d\t%d\t%d\n", B, 2, failed);
    return failed ? 1 : 0;
}
} // namespace
static int crispembed_test_main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "UltraBERT public API: %s\n", e.what());
        return 2;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
