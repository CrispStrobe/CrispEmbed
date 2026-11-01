#include "lama_graph.h"
#include "lama_schema.h"
#include "inpaint_graph.h"
#include "inpaint_graph_spectral.h"
#include "core/env_gate.h"
#include <stdexcept>

namespace lama {
namespace {
using inpaint_graph::Builder;
using latent_inpaint::Tensor;
struct Features {
    ggml_tensor * local = nullptr;
    ggml_tensor * global = nullptr;
};
// lama-v2 files (tools/quantize.cpp): vectors and the first/last conv are F32
// (checked again here). By default every other stored kernel is dequantized to
// F32 at load, so the graph is the F32 graph. CRISPEMBED_LAMA_QUANT_MATMUL=1
// keeps F16 4D kernels and [IC*KH*KW, OC] patch-row quantized kernels resident
// as the ggml_mul_mat src0; transposed and 256-row K-quant kernels stay F32.
struct LamaStorage final : inpaint_graph::StoragePolicy {
    bool matmul = core_env::on("CRISPEMBED_LAMA_QUANT_MATMUL");
    bool keep_stored(const std::string & name, ggml_type type, const int64_t * ne,
                     const std::vector<int64_t> & shape) const override {
        const bool kernel = shape.size() == 4;
        if (!kernel || name == "generator.model.1.ffc.convl2l.weight" || name == "generator.model.34.weight")
            throw std::runtime_error("Big-LaMa graph: tensor must be F32 in its original shape: " + name);
        switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            break;
        default:
            throw std::runtime_error("Big-LaMa graph: unsupported stored type: " + name);
        }
        const bool transpose = name == "generator.model.24.weight" || name == "generator.model.27.weight" ||
                               name == "generator.model.30.weight";
        const bool exact = ne[0] == shape[3] && ne[1] == shape[2] && ne[2] == shape[1] && ne[3] == shape[0];
        const bool patch_rows = ne[0] == shape[1] * shape[2] * shape[3] && ne[1] == shape[0];
        return matmul && !transpose && ((type == GGML_TYPE_F16 && exact) || (type != GGML_TYPE_F32 && patch_rows));
    }
};

ggml_tensor * reflected_conv(Builder & b, const std::string & name, ggml_tensor * x, int stride, int padding) {
    return b.conv(name, padding ? b.reflect_pad(x, padding) : x, stride, 0);
}
ggml_tensor * spectral(Builder & b, const std::string & name, ggml_tensor * input) {
    ggml_tensor * x = ggml_relu(b.ctx, b.batch_norm(name + ".conv1.1", b.conv(name + ".conv1.0", input)));
    ggml_tensor * packed = inpaint_graph::rfft2_ortho(b, x);
    b.mark(name + ".fu.rfft", packed);
    packed = ggml_relu(b.ctx, b.batch_norm(name + ".fu.bn", b.conv(name + ".fu.conv_layer", packed)));
    b.mark(name + ".fu.spectral", packed);
    ggml_tensor * frequency = inpaint_graph::irfft2_ortho(b, packed, int(x->ne[0]));
    b.mark(name + ".fu.irfft", frequency);
    return b.conv(name + ".conv2", ggml_add(b.ctx, x, frequency));
}
Features ffc(Builder & b, const std::string & name, Features input, int stride, int padding) {
    const std::string prefix = name + ".ffc";
    Features out;
    out.local = reflected_conv(b, prefix + ".convl2l", input.local, stride, padding);
    if (input.global)
        out.local = ggml_add(b.ctx, out.local, reflected_conv(b, prefix + ".convg2l", input.global, stride, padding));
    if (b.has(prefix + ".convl2g.weight")) {
        out.global = reflected_conv(b, prefix + ".convl2g", input.local, stride, padding);
        if (input.global) out.global = ggml_add(b.ctx, out.global, spectral(b, prefix + ".convg2g", input.global));
    }
    out.local = ggml_relu(b.ctx, b.batch_norm(name + ".bn_l", out.local));
    if (out.global) out.global = ggml_relu(b.ctx, b.batch_norm(name + ".bn_g", out.global));
    b.mark(name + ".local", out.local);
    if (out.global) b.mark(name + ".global", out.global);
    return out;
}
ggml_tensor * forward(Builder & b, ggml_tensor * input) {
    b.mark("input", input);
    Features x = ffc(b, "generator.model.1", { b.reflect_pad(input, 3), nullptr }, 1, 0);
    for (int block = 2; block <= 4; ++block) x = ffc(b, "generator.model." + std::to_string(block), x, 2, 1);
    for (int block = 5; block <= 22; ++block) {
        const std::string name = "generator.model." + std::to_string(block);
        Features r = ffc(b, name + ".conv1", x, 1, 1);
        r = ffc(b, name + ".conv2", r, 1, 1);
        x.local = ggml_add(b.ctx, x.local, r.local);
        x.global = ggml_add(b.ctx, x.global, r.global);
        b.mark(name + ".local", x.local);
        b.mark(name + ".global", x.global);
    }
    ggml_tensor * out = ggml_concat(b.ctx, x.local, x.global, 2);
    b.mark("generator.model.23", out);
    for (int block : { 24, 27, 30 }) {
        out = b.conv_transpose("generator.model." + std::to_string(block), out, 2, 1, 1);
        out = ggml_relu(b.ctx, b.batch_norm("generator.model." + std::to_string(block + 1), out));
        b.mark("generator.model." + std::to_string(block + 2), out);
    }
    out = ggml_sigmoid(b.ctx, b.conv("generator.model.34", b.reflect_pad(out, 3)));
    b.mark("prediction", out);
    return out;
}
} // namespace

struct GraphModel::Impl {
    LamaStorage storage;
    inpaint_graph::Device device;
    inpaint_graph::Weights weights;
    std::unique_ptr<Builder> graph;
    ggml_tensor *input = nullptr, *output = nullptr;
    int width = 0, height = 0;
#ifdef CRISPEMBED_LAMA_TESTING
    Observer observer = nullptr;
    void * opaque = nullptr;
    static void observe(void * self, const std::string & name, ggml_tensor * tensor,
                        const std::vector<float> & values) {
        auto & p = *static_cast<Impl *>(self);
        Tensor value(int(tensor->ne[2]), int(tensor->ne[1]), int(tensor->ne[0]));
        value.v = values;
        if (p.observer) p.observer(p.opaque, name, value);
    }
#endif
    Impl(const char * path, int threads) : device(threads), weights(device, path, detail::schema(), &storage) {}
    void build(int w, int h) {
        // One geometry cache bounds resident memory for arbitrary-size batches.
        graph.reset();
        graph = std::make_unique<Builder>(device, &weights, 131072);
#ifdef CRISPEMBED_LAMA_TESTING
        graph->set_observer(observer ? observe : nullptr, this);
#endif
        input = graph->parameter("lama.input", { w, h, 4, 1 });
        output = forward(*graph, input);
        graph->finish({ output }, "lama-generator");
        width = w;
        height = h;
    }
};
GraphModel::GraphModel(const char * path, int threads) : impl_(new Impl(path, threads)) {}
GraphModel::~GraphModel() = default;
Tensor GraphModel::run(const Tensor & input) {
    auto & p = *impl_;
    if (!p.graph || p.width != input.w || p.height != input.h) {
        try {
            p.build(input.w, input.h);
        } catch (...) {
            p.graph.reset();
            throw;
        }
    }
    p.graph->upload(p.input, input.v);
    p.graph->compute();
    Tensor result(3, input.h, input.w);
    result.v = p.graph->download(p.output);
    if (!latent_inpaint::finite(result)) throw std::runtime_error("LaMa graph produced nonfinite RGB");
    return result;
}
#ifdef CRISPEMBED_LAMA_TESTING
void GraphModel::set_observer(Observer callback, void * opaque) {
    if (impl_->observer != callback || impl_->opaque != opaque) impl_->graph.reset();
    impl_->observer = callback;
    impl_->opaque = opaque;
}
#endif
} // namespace lama
