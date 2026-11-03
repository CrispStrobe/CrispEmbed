// Hermetic regression test for two encoder tokenizer defects (3 November 2026):
//
//  1. Inputs longer than the learned position table aborted with
//     GGML_ASSERT(i01 >= 0 && i01 < ne01) in get_rows instead of truncating.
//     XLM-R tables have max_position_embeddings = 514 rows but positions start
//     at 2, so only 512 tokens fit; the tokenizers truncated at 514.
//     crispembed.cpp now lowers the tokenizer limit with
//     position_table_token_limit() (gate CRISPEMBED_POS_EMBD_TRUNCATE).
//  2. Cross-encoder pairs for the RoBERTa/XLM-R family were built as
//     <s> a </s> b </s>; HF builds <s> a </s></s> b </s>
//     (gate CRISPEMBED_PAIR_ROBERTA_SEP).
#include "core/clean_exit.h"
#include "tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(cond, ...)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fails++;                                                                                                   \
            printf("FAIL %s: ", #cond);                                                                                \
            printf(__VA_ARGS__);                                                                                       \
            printf("\n");                                                                                              \
        }                                                                                                              \
    } while (0)

static int real_len(const embed_tokens & t) {
    int n = 0;
    for (int m : t.attn_mask) n += m ? 1 : 0;
    return n;
}

static int crispembed_test_main() {
    // position_table_token_limit
    CHECK(position_table_token_limit(514, 2, 514) == 512, "xlmr 514/2");
    CHECK(position_table_token_limit(512, 0, 512) == 512, "bert unchanged");
    CHECK(position_table_token_limit(8194, 2, 8194) == 8192, "bge-m3 8194/2");
    CHECK(position_table_token_limit(514, 2, 512) == 512, "already fits");
    CHECK(position_table_token_limit(0, 2, 512) == 512, "unknown table");

    // SentencePiece (XLM-R ids: <s>=0 <pad>=1 </s>=2 <unk>=3)
    std::vector<std::string> vocab = { "<s>",
                                       "<pad>",
                                       "</s>",
                                       "<unk>",
                                       "\xe2\x96\x81"
                                       "a",
                                       "\xe2\x96\x81"
                                       "b",
                                       "\xe2\x96\x81"
                                       "c",
                                       "\xe2\x96\x81"
                                       "d" };
    SentencePieceTokenizer sp;
    sp.load(vocab, std::vector<float>(vocab.size(), -1.0f), 0, 2, 3, 1, 514);
    sp.set_max_length(position_table_token_limit(514, 2, sp.max_length()));
    CHECK(sp.max_length() == 512, "set_max_length %d", sp.max_length());
    std::string long_text;
    for (int i = 0; i < 1000; i++) long_text += "a ";
    embed_tokens e = sp.encode(long_text);
    CHECK(real_len(e) == 512, "single truncation keeps 512 tokens, got %d", real_len(e));
    CHECK(e.ids.front() == 0 && e.ids[511] == 2, "keeps <s> ... </s> (%d ... %d)", e.ids.front(), e.ids[511]);

    // Pair layout: HF XLMRobertaTokenizer("a b", "c d") = <s> ▁a ▁b </s> </s> ▁c ▁d </s>
    sp.set_pair_double_sep(true);
    embed_tokens p = sp.encode_pair("a b", "c d");
    std::vector<int32_t> want = { 0, 4, 5, 2, 2, 6, 7, 2 };
    std::vector<int32_t> got(p.ids.begin(), p.ids.begin() + real_len(p));
    CHECK(got == want, "double-separator pair layout (len %zu)", got.size());
    sp.set_pair_double_sep(false);
    p = sp.encode_pair("a b", "c d");
    CHECK(real_len(p) == 7 && p.ids[4] == 6, "single-separator legacy layout");

    // Long pair: longest-first truncation to 512 with both separators kept.
    sp.set_pair_double_sep(true);
    p = sp.encode_pair("a b", long_text);
    CHECK(real_len(p) == 512, "pair truncation 512, got %d", real_len(p));
    CHECK(p.ids[3] == 2 && p.ids[4] == 2 && p.ids[511] == 2, "pair keeps </s></s> and final </s>");

    // WordPiece limit setter
    WordPieceTokenizer wp;
    std::vector<std::string> wv = { "[PAD]", "[UNK]", "[CLS]", "[SEP]", "a" };
    wp.load(wv, 2, 3, 1, 0, 514, true);
    wp.set_max_length(position_table_token_limit(514, 2, wp.max_length()));
    embed_tokens w = wp.encode(long_text);
    CHECK(real_len(w) == 512 && w.ids.front() == 2 && w.ids[511] == 3, "wordpiece truncation %d", real_len(w));

    printf("pair-truncation: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
