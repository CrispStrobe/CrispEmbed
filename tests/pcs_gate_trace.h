// Test-driver observer only. Never included by the production PCS source.
#pragma once

#include "core/env_gate.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace pcs_gate_trace {
struct writer {
    FILE * fp = nullptr;
    int request_id = -1, next_offset = 0, offset = 0;

    writer() {
        const char * path = std::getenv("PCS_GATE_CAPTURE_PATH");
        if (path && *path) fp = std::fopen(path, "wx");
        if (!fp) {
            std::fprintf(stderr, "PCS diagnostic capture unavailable (missing path or existing/unwritable file)\n");
            return;
        }
        std::fprintf(fp, "build\t%ld\t", long(__cplusplus));
        hex(__VERSION__);
        std::fprintf(fp, "\n");
#if defined(__AVX2__)
        std::fprintf(fp, "isa\tAVX2\n");
#endif
#if defined(__FMA__)
        std::fprintf(fp, "isa\tFMA\n");
#endif
#if defined(__F16C__)
        std::fprintf(fp, "isa\tF16C\n");
#endif
#if defined(__AVX512F__)
        std::fprintf(fp, "isa\tAVX512F\n");
#endif
#if defined(__AVX512VNNI__)
        std::fprintf(fp, "isa\tAVX512VNNI\n");
#endif
#if defined(__AVXVNNI__)
        std::fprintf(fp, "isa\tAVXVNNI\n");
#endif
        std::fflush(fp);
    }

    void hex(const std::string & value) {
        for (unsigned char c : value) std::fprintf(fp, "%02x", unsigned(c));
    }
    static unsigned bits(float value) {
        static_assert(sizeof(float) == sizeof(uint32_t), "capture requires 32-bit float");
        uint32_t out;
        std::memcpy(&out, &value, sizeof(out));
        return unsigned(out);
    }
};

inline writer * active() {
    if (!core_env::on("PCS_GATE_CAPTURE")) return nullptr;
    static writer out;
    return out.fp ? &out : nullptr;
}

inline void request(const std::string & input, const std::vector<int> & ids, const std::vector<std::string> & pieces) {
    auto * w = active();
    if (!w) return;
    ++w->request_id;
    w->next_offset = 0;
    std::fprintf(w->fp, "request\t%d\t%zu\t", w->request_id, ids.size());
    w->hex(input);
    std::fputc('\n', w->fp);
    for (size_t i = 0; i < ids.size(); ++i) {
        std::fprintf(w->fp, "token\t%d\t%zu\t%d\t", w->request_id, i, ids[i]);
        if (ids[i] >= 0 && size_t(ids[i]) < pieces.size()) w->hex(pieces[ids[i]]);
        std::fputc('\n', w->fp);
    }
    std::fflush(w->fp);
}

inline void chunk(int count) {
    auto * w = active();
    if (!w) return;
    w->offset = w->next_offset;
    w->next_offset += count;
    std::fprintf(w->fp, "chunk\t%d\t%d\t%d\n", w->request_id, w->offset, count);
    std::fflush(w->fp);
}

inline void segmentation(int token, float logit0, float logit1, float probability, bool threshold, bool argmax) {
    auto * w = active();
    if (!w) return;
    std::fprintf(w->fp, "seg\t%d\t%d\t%08x\t%08x\t%08x\t%d\t%d\n", w->request_id, w->offset + token,
                 writer::bits(logit0), writer::bits(logit1), writer::bits(probability), int(threshold), int(argmax));
    std::fflush(w->fp);
}

inline void casing(int token, int slot, float logit, bool bit, bool sentence_initial) {
    auto * w = active();
    if (!w) return;
    std::fprintf(w->fp, "cap\t%d\t%d\t%d\t%08x\t%d\t%d\n", w->request_id, w->offset + token, slot, writer::bits(logit),
                 int(bit), int(sentence_initial));
    std::fflush(w->fp);
}
} // namespace pcs_gate_trace
