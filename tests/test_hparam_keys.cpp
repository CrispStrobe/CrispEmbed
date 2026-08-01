// Unit test for architecture-driven GGUF hparam key resolution (A2 / issue #33).
//
// Host-only: no model, no ggml. Drives src/core/hparam_keys.h against fake key
// sets that mirror real GGUFs:
//   - nomic-embed-text-v2-moe (llama.cpp export): ONLY nomic-bert-moe.* keys.
//     Before A2 the loader read bert.*/xlmr.* only, silently fell back to
//     384-dim/6-layer, and would have emitted a garbage embedding.
//   - a CrispEmbed-converted bert GGUF: bert.* keys — must be unaffected.
//
// Also pins the A/B contract for CRISPEMBED_ARCH_HPARAMS: with arch-derived
// lookup disabled, resolution must be exactly the legacy behaviour.
//
// Returns non-zero exit code on any failure.

#include "core/clean_exit.h"
#include "core/hparam_keys.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(const char * name, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) g_failures++;
}

// A fake GGUF key/value table + a lookup matching core_hparams::resolve()'s contract.
using KV = std::map<std::string, int>;
static auto make_lookup(const KV & kv) {
    return [&kv](const std::string & key, int & out) -> bool {
        auto it = kv.find(key);
        if (it == kv.end()) return false;
        out = it->second;
        return true;
    };
}

// The real candidate order used by load_model() for n_embd / n_layer.
static std::vector<std::string> embd_keys(const std::string & arch, bool on) {
    return { "bert.hidden_size", "bert.embedding_length", "xlmr.embedding_length",
             core_hparams::arch_key(arch, "embedding_length", on), core_hparams::arch_key(arch, "hidden_size", on) };
}
static std::vector<std::string> layer_keys(const std::string & arch, bool on) {
    return { "bert.num_hidden_layers", "bert.block_count", "xlmr.block_count",
             core_hparams::arch_key(arch, "block_count", on) };
}

static int crispembed_test_main() {
    std::printf("test_hparam_keys\n");

    // ---- arch_key() ----
    check("arch_key builds <arch>.<field>",
          core_hparams::arch_key("nomic-bert-moe", "embedding_length", true) == "nomic-bert-moe.embedding_length");
    check("arch_key disabled -> empty", core_hparams::arch_key("nomic-bert-moe", "embedding_length", false).empty());
    check("arch_key empty arch -> empty", core_hparams::arch_key("", "embedding_length", true).empty());
    check("arch_key null field -> empty", core_hparams::arch_key("bert", nullptr, true).empty());

    // ---- resolve(): skips empty keys, honours order ----
    {
        KV kv = { { "b", 2 }, { "c", 3 } };
        int v = -1;
        check("resolve finds first present (order)",
              core_hparams::resolve(make_lookup(kv), { "a", "b", "c" }, v) && v == 2);
        v = -1;
        check("resolve skips empty keys", core_hparams::resolve(make_lookup(kv), { "", "c" }, v) && v == 3);
        v = -1;
        check("resolve reports miss", !core_hparams::resolve(make_lookup(kv), { "x", "" }, v) && v == -1);
        check("resolve leaves out untouched on miss", v == -1);
    }

    // ---- Real case: nomic-embed-text-v2-moe (only nomic-bert-moe.* present) ----
    {
        const std::string arch = "nomic-bert-moe";
        KV kv = {
            { "nomic-bert-moe.embedding_length", 768 },    { "nomic-bert-moe.block_count", 12 },
            { "nomic-bert-moe.attention.head_count", 12 }, { "nomic-bert-moe.feed_forward_length", 3072 },
            { "nomic-bert-moe.expert_count", 8 },          { "nomic-bert-moe.expert_used_count", 2 },
            { "nomic-bert-moe.pooling_type", 1 },
        };
        int embd = 384, layer = 6; // the silent defaults
        bool f1 = core_hparams::resolve(make_lookup(kv), embd_keys(arch, true), embd);
        bool f2 = core_hparams::resolve(make_lookup(kv), layer_keys(arch, true), layer);
        check("nomic: n_embd resolves to 768 (not the 384 default)", f1 && embd == 768);
        check("nomic: n_layer resolves to 12 (not the 6 default)", f2 && layer == 12);

        // A/B: with the gate OFF this is exactly the legacy behaviour — miss +
        // silent default. This is the bug A2 exists to remove.
        int embd_off = 384, layer_off = 6;
        bool g1 = core_hparams::resolve(make_lookup(kv), embd_keys(arch, false), embd_off);
        bool g2 = core_hparams::resolve(make_lookup(kv), layer_keys(arch, false), layer_off);
        check("A/B gate off: nomic n_embd MISSES (legacy behaviour)", !g1 && embd_off == 384);
        check("A/B gate off: nomic n_layer MISSES (legacy behaviour)", !g2 && layer_off == 6);
        check("A/B: miss is detectable -> strict mode can hard-fail", !g1);
    }

    // ---- Regression: a CrispEmbed-converted bert GGUF is unaffected either way ----
    {
        const std::string arch = "bert";
        KV kv = { { "bert.hidden_size", 384 }, { "bert.num_hidden_layers", 6 } };
        int e_on = -1, e_off = -1, l_on = -1, l_off = -1;
        core_hparams::resolve(make_lookup(kv), embd_keys(arch, true), e_on);
        core_hparams::resolve(make_lookup(kv), embd_keys(arch, false), e_off);
        core_hparams::resolve(make_lookup(kv), layer_keys(arch, true), l_on);
        core_hparams::resolve(make_lookup(kv), layer_keys(arch, false), l_off);
        check("A/B: bert GGUF identical with gate on/off (n_embd)", e_on == 384 && e_off == 384);
        check("A/B: bert GGUF identical with gate on/off (n_layer)", l_on == 6 && l_off == 6);
    }

    // ---- Legacy keys win over arch keys (precedence is stable) ----
    {
        const std::string arch = "bert";
        KV kv = { { "bert.hidden_size", 111 }, { "bert.embedding_length", 222 } };
        int v = -1;
        core_hparams::resolve(make_lookup(kv), embd_keys(arch, true), v);
        check("precedence: bert.hidden_size wins over bert.embedding_length", v == 111);
    }

    // ---- An unknown future arch resolves with no new code (the whole point) ----
    {
        const std::string arch = "jina-bert-v2";
        KV kv = { { "jina-bert-v2.embedding_length", 512 }, { "jina-bert-v2.block_count", 4 } };
        int e = 384, l = 6;
        check("future arch: n_embd resolves generically",
              core_hparams::resolve(make_lookup(kv), embd_keys(arch, true), e) && e == 512);
        check("future arch: n_layer resolves generically",
              core_hparams::resolve(make_lookup(kv), layer_keys(arch, true), l) && l == 4);
    }

    std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
