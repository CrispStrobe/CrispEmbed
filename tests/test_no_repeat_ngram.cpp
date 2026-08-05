// test_no_repeat_ngram.cpp — hermetic guard for core/no_repeat_ngram.h
// (F1: the deepseek-ocr2 repetition guard; also the qwen2vl/internvl2
// ngram=3 guard). No weights, no GGUF, no network.
//
// The semantics under guard are transformers' NoRepeatNGramLogitsProcessor:
// ban exactly the tokens that would complete an ngram already present in the
// history. The property that matters in production is the last case below —
// with the guard, a greedy loop whose logits keep proposing the same phrase
// cannot repeat it forever (the T14 finding: 2 of 5 cc0 gold pages spiralled
// "FinlandFinland..." into the 1024-token cap; the reference contract
// generates with no_repeat_ngram_size=20).
#include "core/no_repeat_ngram.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <vector>

static int g_checks = 0, g_failures = 0;

static void expect_eq(int got, int want, const char * what) {
    g_checks++;
    if (got != want) {
        g_failures++;
        fprintf(stderr, "FAIL %s: got %d want %d\n", what, got, want);
    }
}

// Logits where token `top` scores highest, then descending by index distance.
static std::vector<float> favor(int V, int top) {
    std::vector<float> l(V);
    for (int v = 0; v < V; v++) l[v] = 10.0f - (float)((v - top + V) % V);
    return l;
}

static int crispembed_test_main() {
    using core_decode::argmax_no_repeat_ngram;
    const int V = 8;

    // Empty history / short history / ngram<=1: plain argmax.
    {
        std::vector<int32_t> h;
        auto l = favor(V, 3);
        expect_eq(argmax_no_repeat_ngram(l.data(), V, h, 3), 3, "empty history");
        expect_eq(argmax_no_repeat_ngram(l.data(), V, h, 0), 3, "ngram=0 is plain argmax");
        expect_eq(argmax_no_repeat_ngram(l.data(), V, h, 1), 3, "ngram=1 is plain argmax");
        h = { 1 }; // shorter than k=2
        expect_eq(argmax_no_repeat_ngram(l.data(), V, h, 3), 3, "history shorter than ngram-1");
    }

    // ngram=3 over [1,2,3,1,2]: suffix [1,2] occurred at i=0 followed by 3,
    // so 3 is banned; the argmax favouring 3 must pick its runner-up 4.
    {
        std::vector<int32_t> h = { 1, 2, 3, 1, 2 };
        auto l = favor(V, 3);
        expect_eq(argmax_no_repeat_ngram(l.data(), V, h, 3), 4, "bans the repeating continuation");
        // A different top token is unaffected.
        auto l5 = favor(V, 5);
        expect_eq(argmax_no_repeat_ngram(l5.data(), V, h, 3), 5, "non-banned top unaffected");
        // ngram=4 (suffix [3,1,2] never occurred before): no ban.
        expect_eq(argmax_no_repeat_ngram(l.data(), V, h, 4), 3, "longer ngram no match no ban");
    }

    // Multiple occurrences ban multiple continuations.
    {
        // suffix k=2 is [1,2]; [1,2]->3 at i=0 and [1,2]->5 at i=3.
        std::vector<int32_t> h = { 1, 2, 3, 1, 2, 5, 1, 2 };
        auto l3 = favor(V, 3);
        // 3 banned -> runner-up 4.
        expect_eq(argmax_no_repeat_ngram(l3.data(), V, h, 3), 4, "first continuation banned");
        auto l5 = favor(V, 5);
        // 5 banned; favor(V,5) order after 5 is 6.
        expect_eq(argmax_no_repeat_ngram(l5.data(), V, h, 3), 6, "second continuation banned");
    }

    // int (not int32_t) history instantiates too — the qwen2vl/internvl2 type.
    {
        std::vector<int> h = { 1, 2, 3, 1, 2 };
        auto l = favor(V, 3);
        expect_eq(argmax_no_repeat_ngram(l.data(), V, h, 3), 4, "std::vector<int> instantiation");
    }

    // All-banned fallback: V=1, ngram=2, history [0,0] bans token 0 — the
    // fallback returns the unconstrained argmax instead of -1.
    {
        std::vector<int32_t> h = { 0, 0 };
        float l1[1] = { 0.5f };
        expect_eq(argmax_no_repeat_ngram(l1, 1, h, 2), 0, "all-banned falls back to argmax");
    }

    // The production property: a greedy loop that keeps favouring the same
    // 3-token phrase spirals forever unguarded, and must be broken by the
    // guard within one extra phrase length. Simulated: logits always favour
    // continuing the cycle [6,7,2].
    {
        const int cycle[3] = { 6, 7, 2 };
        std::vector<int32_t> h;
        int broke_at = -1;
        for (int step = 0; step < 64; step++) {
            auto l = favor(V, cycle[step % 3]);
            int pick = argmax_no_repeat_ngram(l.data(), V, h, 3);
            h.push_back(pick);
            if (pick != cycle[step % 3]) {
                broke_at = step;
                break;
            }
        }
        g_checks++;
        // One full cycle emits [6,7,2]; the second pass may emit 6,7 again
        // (suffix pairs not yet repeated) but [7,2]->6 and successors are then
        // banned — the loop must break within the second pass (step <= 5).
        if (broke_at < 0 || broke_at > 5) {
            g_failures++;
            fprintf(stderr, "FAIL guarded greedy loop: broke_at=%d (want 0..5)\n", broke_at);
        }
        // And the unguarded loop provably does NOT break (the spiral).
        std::vector<int32_t> h2;
        bool spiralled = true;
        for (int step = 0; step < 64; step++) {
            auto l = favor(V, cycle[step % 3]);
            int pick = argmax_no_repeat_ngram(l.data(), V, h2, 0); // guard off
            h2.push_back(pick);
            if (pick != cycle[step % 3]) spiralled = false;
        }
        g_checks++;
        if (!spiralled) {
            g_failures++;
            fprintf(stderr, "FAIL unguarded loop unexpectedly broke the cycle\n");
        }
    }

    printf("no-repeat-ngram: %d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

// tools/check_test_clean_exit.sh: a one-shot binary must not run ggml's
// static GPU-device destructor at exit (it aborts on Metal / faults on CUDA).
int main() {
    core_util::clean_exit(crispembed_test_main());
}
