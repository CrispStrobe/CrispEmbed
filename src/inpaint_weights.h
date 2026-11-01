// Shared loader for crispembed-quantize "inpaint-v1" files (MI-GAN, Moebius,
// PixelHacker, SDXL VAE). F32 files keep each engine's original load path
// unchanged; this helper is used only for policy files, behind the default-off
// CRISPEMBED_INPAINT_QUANT gate.
//
// Policy layout (tools/quantize.cpp inpaint_tensor_plan): every tensor is
// either F32 in its original shape, or stored as a 2D natural-row matrix
// [IC*KH*KW, OC] (conv) / [IN, OUT] (linear) in F16 or a ggml block type.
// Default: non-F32 tensors are dequantized at load into a second CPU buffer
// with their original shape, so the engines run their unchanged F32 graphs.
// CRISPEMBED_INPAINT_QUANT_MATMUL=1 instead keeps them in their stored type
// for direct ggml_mul_mat (1x1/kxk convs with groups 1 and linear layers).
#pragma once

#include "core/gguf_loader.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>

namespace inpaint_weights {

using Shape = std::array<int64_t, 4>; // ggml ne order
using ShapeMap = std::map<std::string, Shape>;

constexpr const char * policy_key = "inpaint.quantization.policy";
constexpr const char * policy_name = "inpaint-v1";

struct Stats {
    int resident = 0, dequantized = 0, stored = 0; // stored = kept quantized/F16 for matmul
    size_t weight_bytes = 0;
    std::map<std::string, int> stored_types; // ggml type name -> tensor count in the file
};

struct Converted {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    Converted() = default;
    Converted(const Converted &) = delete;
    Converted & operator=(const Converted &) = delete;
    ~Converted();
};

// True when the file declares the inpaint-v1 policy (a precision variant).
bool is_policy_file(gguf_context * meta);
// For a policy file: the CRISPEMBED_INPAINT_QUANT gate is on and every
// expected tensor is F32-original-shape or a supported 2D natural-row type.
bool validate(gguf_context * meta, const ShapeMap & expected, const char * tag, std::string & error);
// Loads a validated policy file. On success weights.tensors maps every name
// to an F32 original-shape tensor, except stored matmul tensors when
// keep_stored (CRISPEMBED_INPAINT_QUANT_MATMUL) is on.
bool load(const char * path, gguf_context * meta, ggml_backend_t backend, const ShapeMap & expected, bool keep_stored,
          const char * tag, core_gguf::WeightLoad & weights, Converted & converted, Stats & stats, std::string & error);
bool quant_gate();
bool matmul_gate();

} // namespace inpaint_weights
