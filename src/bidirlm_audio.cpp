// bidirlm_audio.cpp — thin wrapper that adapts crisp_audio for BidirLM-Omni
// in CrispEmbed.
//
// crisp_audio (from ../CrispASR/crisp_audio) does the heavy lifting: log-mel,
// conv stem, transformer encoder, projection. This file:
//
//   1. Recognizes a BidirLM-Omni audio GGUF at load time.
//   2. Calls crisp_audio_init_from_file with the right tensor/meta prefix.
//   3. Translates crisp_audio_encode's per-frame output (n_frames, output_dim)
//      into a single L2-normalized embedding via mean pooling, matching the
//      `model.encode(audio)` behavior of sentence-transformers.
//
// Built only when CRISPEMBED_HAS_CRISP_AUDIO is defined (gated by CMake).

#include "crispembed.h"

#ifdef CRISPEMBED_HAS_CRISP_AUDIO
#include "crisp_audio.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace bidirlm_audio {

struct context {
    crisp_audio_context* ca = nullptr;
    int output_dim = 0;
    std::vector<float> last_pooled;  // owned buffer returned to caller

    ~context() {
        if (ca) crisp_audio_free(ca);
    }
};

context* open(const char* gguf_path, int n_threads, bool use_gpu) {
    crisp_audio_params p = crisp_audio_params_default();
    p.n_threads = n_threads;
    p.use_gpu = use_gpu;
    // BidirLM-Omni's converter writes audio tensors under "audio_tower." and
    // metadata under "bidirlm.audio.". crisp_audio also has a fallback to
    // qwen3-asr's keys when the BidirLM ones aren't present, so passing the
    // BidirLM prefix here is harmless even on a qwen3-asr GGUF.
    p.tensor_prefix = "audio_tower.";
    p.meta_prefix   = "bidirlm.audio.";
    p.dialect       = CRISP_AUDIO_DIALECT_QWEN_OMNI;

    crisp_audio_context* ca = crisp_audio_init_from_file(gguf_path, &p);
    if (!ca) return nullptr;

    auto* ctx = new context();
    ctx->ca = ca;
    ctx->output_dim = crisp_audio_output_dim(ca);
    return ctx;
}

const float* encode(context* ctx, const float* pcm, int n_samples,
                    int* out_dim) {
    if (!ctx || !ctx->ca || !pcm || n_samples <= 0) return nullptr;

    int n_mels = 0, T_mel = 0;
    float* mel = crisp_audio_compute_mel(ctx->ca, pcm, n_samples,
                                         &n_mels, &T_mel);
    if (!mel) return nullptr;

    int n_frames = 0, dim = 0;
    float* enc = crisp_audio_encode(ctx->ca, mel, n_mels, T_mel,
                                    &n_frames, &dim);
    std::free(mel);
    if (!enc || n_frames <= 0 || dim <= 0) {
        std::free(enc);
        return nullptr;
    }

    // Mean pool over frames, then L2 normalize. Matches sentence-transformers
    // Pooling("mean") + similarity normalization for the audio modality.
    ctx->last_pooled.assign(dim, 0.0f);
    for (int f = 0; f < n_frames; f++) {
        const float* row = enc + (size_t)f * dim;
        for (int i = 0; i < dim; i++) ctx->last_pooled[i] += row[i];
    }
    const float inv = 1.0f / (float)n_frames;
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        ctx->last_pooled[i] *= inv;
        norm_sq += ctx->last_pooled[i] * ctx->last_pooled[i];
    }
    const float norm = std::sqrt(std::max(norm_sq, 1e-12f));
    for (int i = 0; i < dim; i++) ctx->last_pooled[i] /= norm;

    std::free(enc);
    if (out_dim) *out_dim = dim;
    return ctx->last_pooled.data();
}

void close(context* ctx) { delete ctx; }

}  // namespace bidirlm_audio

#endif  // CRISPEMBED_HAS_CRISP_AUDIO
