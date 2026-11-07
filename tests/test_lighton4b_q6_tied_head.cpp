#include "qwen35_decoder.h"
#include "qwen35_math.h"
#include "lighton4b_bf16_dequant_weights.h"
#include "core/clean_exit.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void gate(const char * n, const char * v) {
#ifdef _WIN32
    _putenv_s(n, v ? v : "");
#else
    if (v)
        setenv(n, v, 1);
    else
        unsetenv(n);
#endif
}
static float rne(float x) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    b += 0x7fff + ((b >> 16) & 1);
    b &= 0xffff0000;
    std::memcpy(&x, &b, 4);
    return x;
}
static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * 4);
}
static int run() {
    using namespace qwen35_ocr;
    constexpr int width = 256, rows = 257, tokens = 3;
    int checks = 0;
    auto check = [&](bool ok) {
        ++checks;
        if (!ok) {
            std::fprintf(stderr, "Q6_TIED_HEAD failed=%d\n", checks);
            std::exit(1);
        }
    };
    auto * ctx = ggml_init({ 65536, nullptr, true });
    auto * backend = ggml_backend_cpu_init();
    check(ctx && backend);
    auto * embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, width, rows);
    auto * head = embedding;
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer && ggml_type_size(GGML_TYPE_Q6_K) == 210);
    std::vector<uint8_t> raw_bytes(ggml_nbytes(embedding));
    std::vector<float> reference(width * rows), rounded(reference.size());
    const uint16_t dbits[] = { 0x3800, 0xb400, 0 };
    const float scales[] = { .5f, -.25f, 0 };
    for (int block = 0; block < rows; ++block) {
        auto * b = raw_bytes.data() + 210 * block;
        for (int k = 0; k < 208; ++k) b[k] = uint8_t((block * 37 + k * 13 + 17) % 256);
        std::memcpy(b + 208, &dbits[block % 3], 2);
        // Independent output-index Q6_K bit extraction, signed scales and D.
        for (int k = 0; k < width; ++k) {
            const int half = k / 128, p = k % 128;
            const int lo = (b[half * 64 + p % 64] >> (p >= 64 ? 4 : 0)) & 15;
            const int hi = (b[128 + half * 32 + p % 32] >> (2 * (p / 32))) & 3;
            int s = b[192 + k / 16];
            if (s >= 128) s -= 256;
            reference[block * width + k] = (scales[block % 3] * s) * ((lo | (hi << 4)) - 32);
        }
    }
    ggml_backend_tensor_set(embedding, raw_bytes.data(), 0, raw_bytes.size());
    ggml_set_name(embedding, "l.embed_tokens.weight");
    for (auto * g : { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_QWEN35_QUANT", "CRISPEMBED_QWEN35_STREAM_PROJECTION",
                      "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", "CRISPEMBED_LIGHTON4B_QUANT" })
        gate(g, "1");
    for (size_t k = 0; k < rounded.size(); ++k) rounded[k] = rne(reference[k]);
    check(!same(reference, rounded));
    const std::vector<int> ids = { 0, 127, 256, 1, 256 };
    std::vector<float> got, expected, hidden(width * tokens), scores;
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < width; ++k) hidden[t * width + k] = float((k + 11 * t) % 3 - 1) / 32;
    for (auto * q6 : { static_cast<const char *>(nullptr), "", "0", "1" }) {
        gate("CRISPEMBED_LIGHTON4B_Q6_K", q6);
        for (auto * bf16 : { static_cast<const char *>(nullptr), "", "0", "1" }) {
            gate("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", bf16);
            const bool active = q6 && *q6 == '1' && bf16 && *bf16 == '1';
            const auto & values = active ? rounded : reference;
            expected.clear();
            for (int id : ids)
                expected.insert(expected.end(), values.begin() + id * width, values.begin() + (id + 1) * width);
            check(embed_cpu(embedding, ids, got) && same(got, expected));
            for (const char * mode : { "0", "1" }) {
                gate("CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE", mode);
                for (int tile : { 17, 31, 256 }) {
                    scores = { 42 };
                    const bool ok = project_rows_cpu(head, hidden.data(), width, tokens, 2, tile, scores);
                    if (!q6 || *q6 != '1') {
                        check(!ok && scores.empty());
                        continue;
                    }
                    check(ok && scores.size() == rows * tokens);
                    // Dyadic inputs keep every partial sum exact in F32; no dot-order tolerance.
                    expected.assign(rows * tokens, 0);
                    for (int t = 0; t < tokens; ++t)
                        for (int row = 0; row < rows; ++row) {
                            float sum = 0;
                            for (int k = 0; k < width; ++k) sum += values[row * width + k] * hidden[t * width + k];
                            expected[t * rows + row] = sum;
                        }
                    check(same(scores, expected));
                }
            }
        }
    }
    // Gather rejects both ends outside the matrix and clears stale output.
    for (int invalid : { -1, rows }) {
        got = { 42 };
        check(!embed_cpu(embedding, { invalid }, got) && got.empty());
    }
    gate("CRISPEMBED_QWEN35_QUANT", "0");
    check(!embed_cpu(embedding, { 0 }, got) && got.empty());
    check(!project_rows_cpu(head, hidden.data(), width, tokens, 2, 31, scores) && scores.empty());
    std::vector<uint8_t> after(raw_bytes.size());
    ggml_backend_tensor_get(embedding, after.data(), 0, after.size());
    check(after == raw_bytes && head == embedding);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::printf("Q6_TIED_HEAD passed=1 checks=%d independent_scalar_values=65792 tied_alias=1 gathered_boundary_rows=1 "
                "streamed_odd_tiles=1 bf16_gate=1 storage_unchanged=1 model_payload=0\n",
                checks);
    return 0;
}
int main() {
    core_util::clean_exit(run());
}
