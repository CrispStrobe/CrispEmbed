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

// Reproduce HF's _get_feat_extract_output_lengths: 3 successive
// stride-2 conv kernels with k=3, pad=1 → floor((L-1)/2)+1 each pass.
static int conv_out_len_3x(int t_len) {
    auto step = [](int x) { return (x - 1) / 2 + 1; };
    return step(step(step(t_len)));
}

const float* encode(context* ctx, const float* pcm, int n_samples,
                    int* out_dim) {
    if (!ctx || !ctx->ca || !pcm || n_samples <= 0) return nullptr;

    int n_mels = 0, T_mel = 0;
    float* mel = crisp_audio_compute_mel(ctx->ca, pcm, n_samples,
                                         &n_mels, &T_mel);
    if (!mel) return nullptr;

    int n_frames_padded = 0, dim = 0;
    float* enc = crisp_audio_encode(ctx->ca, mel, n_mels, T_mel,
                                    &n_frames_padded, &dim);
    std::free(mel);
    if (!enc || n_frames_padded <= 0 || dim <= 0) {
        std::free(enc);
        return nullptr;
    }

    // crisp_audio_encode pads each chunk's mel to chunk_T frames and runs
    // the encoder on the full padded sequence. For a BidirLM-style mean
    // pool we need to skip the silence-padded frames at the end of the
    // tail chunk; the HF reference filters them out via padded_mask_after_cnn
    // BEFORE the encoder runs (line 353 of modeling_bidirlm_omni.py), so
    // the cleanest mid-graph alignment is unattainable from here, but
    // skipping them in the pooling loop catches the dominant error term.
    const int n_window = crisp_audio_n_window(ctx->ca);
    const int chunk_T = n_window > 0 ? n_window * 2 : 200;  // BidirLM default
    const int num_chunks = (T_mel + chunk_T - 1) / chunk_T;
    const int T_chunk_out = conv_out_len_3x(chunk_T);

    ctx->last_pooled.assign(dim, 0.0f);
    int n_valid = 0;
    for (int c = 0; c < num_chunks; c++) {
        const int t_start_mel = c * chunk_T;
        const int t_len_mel   = std::min(chunk_T, T_mel - t_start_mel);
        const int valid       = conv_out_len_3x(t_len_mel);
        const int frame_off   = c * T_chunk_out;
        for (int f = 0; f < valid; f++) {
            const float* row = enc + (size_t)(frame_off + f) * dim;
            for (int i = 0; i < dim; i++) ctx->last_pooled[i] += row[i];
        }
        n_valid += valid;
    }
    if (n_valid == 0) n_valid = n_frames_padded;  // safety net

    const float inv = 1.0f / (float)n_valid;
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
