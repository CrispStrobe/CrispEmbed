// unlimited_ocr2_dash.h — Unlimited-OCR-2 (baidu, MIT) decoder pieces that differ
// from v1: page2d rope coordinates, DASH block routing and block-sparse attention.
// Pure CPU/F32 functions; see docs/unlimited-ocr2-inference-spec.md for the
// reference semantics (section numbers below refer to it).
#pragma once
#include <cstdint>
#include <vector>

namespace uocr2 {
constexpr int32_t IMG = 128815, PAGE = 128827;
constexpr int HD = 128, BLOCK = 32, LOCAL_BLOCKS = 4, LOCAL_WINDOW = 128, TOPM = 16, TOPK = 12;
constexpr float SCORE_NEG = -1e9f, ENTMAX_NEG = -1e4f, FLOOR = 1e-4f, SIGMA = 1.0f;

// §2.3/§2.4 rope coordinates, plus the decode state machine driven by fed tokens.
struct coords {
    std::vector<int> tok, page; // per prompt position
    std::vector<int> pf_cnt;    // prefill count per page bucket
    int n_pg = 1;
    int cur_tok = 0, cur_page = 0; // decode state (seeded from the last prompt position)
    bool page_mode = false;        // false => the 1D YaRN fallback applies (not implemented)
    void advance(int32_t fed);     // §2.4
};
coords prefill_coords(const std::vector<int32_t> & ids);

// cos/sin for one (tok, page) coordinate, 128 entries each (split-half duplicated).
void rope_table(int tok, int page, float * cos128, float * sin128);
// NEOX split-half rotation of one 128-dim head vector.
void apply_rope(float * x, const float * cos128, const float * sin128);
// Gemma RMSNorm x*rsqrt(mean(x^2)+eps)*(1+w), clamped to ±1e4 (router q/k norms).
void gemma_rms(float * x, int n, const float * w, float eps = 1e-6f);

// §3.5 / §4.4 page-aware candidate domain bookkeeping.
struct pages {
    std::vector<int> img_page;            // per prompt position, -1 if not an image token
    std::vector<int> q_page;              // per prompt position, prefill query page (-1 = open)
    std::vector<int> bmin, bmax;          // per prefill block: image page range (-1 if none)
    int prompt_pages = 0, mcum = 0, n_prefill_blocks = 0;
    void advance(int32_t fed) { if (fed == PAGE) ++mcum; }
    int decode_qp() const { const int qp = mcum - 1 - prompt_pages; return qp < 0 ? -1 : qp; }
};
pages prefill_pages(const std::vector<int32_t> & ids);

// Per-layer router key store with cached head_cls block summaries (§3.4).
struct router_keys {
    std::vector<float> ik;   // rows * 128
    std::vector<float> kbar; // complete blocks * 128 (immutable once complete)
    int rows() const { return int(ik.size() / HD); }
    void append(const float * row, const float * head_cls);
    // Summary of block c (complete or partial).
    void summary(int c, const float * head_cls, float * out) const;
};

struct selection {
    std::vector<int> blocks;  // valid selected remote blocks (<= 12)
    std::vector<float> bias;  // one bias per selected block
};
// §3.4 scores + §3.6 routing for one query at absolute position pos.
selection route(const router_keys & rk, const float * iq, const float * head_cls, int pos,
                const std::vector<uint8_t> & candidate);
// 1.5-entmax over n values (reference DA:772-800).
void entmax15(const float * x, int n, float * out);

// §3.7 / §4.6: one query row, one head; K/V rows are [pos+1][HD] for that head.
void attend(const float * q, const float * k, const float * v, int kv_stride, int pos, const selection & sel,
            float * out);
} // namespace uocr2
