// Shared, bounded CPU/F32 operators for the experimental latent inpainters.
// Reuses core_gguf ownership, core_cpu normalization and ggml CPU GEMM.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct gguf_context;

namespace latent_inpaint {

struct Tensor {
    int c = 0, h = 0, w = 0;
    std::vector<float> v;
    Tensor() = default;
    Tensor(int channels, int height, int width);
    size_t spatial() const { return size_t(h) * size_t(w); }
    float & at(int channel, int y, int x) { return v[(size_t(channel) * h + y) * w + x]; }
    const float & at(int channel, int y, int x) const { return v[(size_t(channel) * h + y) * w + x]; }
};

struct Weight {
    // Original PyTorch dimensions, except the explicitly documented singleton
    // Conv3d depth collapse. Storage remains contiguous in that order.
    std::vector<int64_t> shape;
    const float * data = nullptr;
    size_t size = 0;
};

using ShapeMap = std::map<std::string, std::vector<int64_t>>;

class Engine {
public:
    Engine(const char * path, int threads, const ShapeMap & schema);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    const Weight & weight(const std::string & name) const;
    bool has(const std::string & name) const;
    Tensor conv(const std::string & name, const Tensor & x, int stride = 1, int pad = 0, int groups = 1,
                int dilation = 1);
    Tensor conv_transpose(const std::string & name, const Tensor & x, int stride = 2, int padding = 1,
                          int output_padding = 1);
    Tensor linear(const std::string & name, const Tensor & x, bool bias = true);
    Tensor group_norm(const std::string & name, const Tensor & x, int groups = 32, float eps = 1e-5f) const;
    Tensor layer_norm(const std::string & name, const Tensor & x, float eps = 1e-5f) const;
    Tensor batch_norm(const std::string & name, const Tensor & x, float eps = 1e-5f) const;
    // Noncausal scaled dot-product attention. F32 Q/K/V and accumulation;
    // ggml's tiled CPU kernel avoids allocating a full spatial score matrix.
    Tensor attention(const Tensor & q, const Tensor & k, const Tensor & v, int heads = 1);
    void mark(const std::string & name, const Tensor & x) const;
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
    using Observer = void (*)(void *, const std::string &, const Tensor &);
    void set_observer(Observer callback, void * opaque);
#endif
private:
    void load_precision_variant(const char * path, int threads, const ShapeMap & schema, gguf_context * meta);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

Tensor add(const Tensor & a, const Tensor & b);
Tensor concat(const Tensor & a, const Tensor & b);
Tensor silu(Tensor x);
Tensor gelu(Tensor x);
Tensor resize_nearest(const Tensor & x, int height, int width);
Tensor pad(const Tensor & x, int left, int right, int top, int bottom);
bool finite(const Tensor & x);

} // namespace latent_inpaint
