// src/core/gguf_loader.h — shared GGUF weight loading scaffolding.
//
// Every model implementation in src/ has its own copy of the "open a
// GGUF file, read its hyperparameters, allocate a backend buffer, mmap
// the weight data, and build a name -> tensor lookup map" dance. The
// code is ~40-60 lines per model and is essentially identical across
// them, with only the model-specific prefix and tensor naming scheme
// changing.
//
// This helper extracts the shared scaffolding. What stays model-specific:
//
//   * Hyperparameter reading (each model has its own hparams struct
//     and GGUF key prefix, e.g. "parakeet.n_layers" vs "voxtral.n_layers").
//   * Vocabulary / tokenizer loading (varies by tokenizer type).
//   * The actual per-field assignment loop that pulls tensors out of
//     the map and stores them in per-layer struct fields.
//
// What this helper does for the model:
//
//   * Opens the GGUF file in two passes (metadata, then tensor alloc).
//   * Provides scalar / string / array reader helpers with defaults.
//   * Allocates the backend buffer and mmap-copies the weight data.
//   * Builds the std::unordered_map<std::string, ggml_tensor *> tensor
//     lookup map and returns it in a WeightLoad struct.
//   * Provides require() / try_get() tensor lookup helpers that log a
//     sensible error message when a required tensor is missing.
//
// Usage pattern (each model's *_model_load function):
//
//   // 1. Metadata pass — read hyperparameters.
//   gguf_context * meta = core_gguf::open_metadata(path);
//   if (!meta) return false;
//   hp.n_layers = core_gguf::kv_u32(meta, "mymodel.n_layers", hp.n_layers);
//   // ... other hparams
//   core_gguf::load_vocab_array(meta, "tokenizer.ggml.tokens", vocab);
//   core_gguf::free_metadata(meta);
//
//   // 2. Weight pass — allocate backend buffer, mmap, build tensor map.
//   core_gguf::WeightLoad wl;
//   if (!core_gguf::load_weights(path, backend, wl)) return false;
//   model.ctx     = wl.ctx;
//   model.buf     = wl.buf;
//   model.tensors = std::move(wl.tensors);
//
//   // 3. Bind named tensors into struct fields.
//   model.attn.q_w = core_gguf::require(model.tensors, "encoder.attn.q.weight", "mymodel");
//   ... etc.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <unordered_map>
#include <string>
#include <unordered_map>
#include <vector>

namespace core_gguf {

// ---------------------------------------------------------------------------
// Pass 1: metadata (hyperparameters + vocab).
// ---------------------------------------------------------------------------

// Open the GGUF for metadata-only reading. Returns a gguf_context owned
// by the caller — free with free_metadata() when done reading keys.
// Returns nullptr and prints an error to stderr on failure.
gguf_context * open_metadata(const char * path);

// Free a gguf_context obtained from open_metadata().
void free_metadata(gguf_context * gctx);

// Scalar key readers with defaults. All return the default value when
// the key is absent or the type doesn't match.
uint32_t kv_u32(gguf_context * gctx, const char * key, uint32_t default_val);
int32_t kv_i32(gguf_context * gctx, const char * key, int32_t default_val);
float kv_f32(gguf_context * gctx, const char * key, float default_val);
bool kv_bool(gguf_context * gctx, const char * key, bool default_val);
std::string kv_str(gguf_context * gctx, const char * key, const char * default_val);

// Read a string array (e.g. tokenizer.ggml.tokens). Returns an empty
// vector when the key is missing or has the wrong type.
std::vector<std::string> kv_str_array(gguf_context * gctx, const char * key);

// Read an int32 array. Returns empty vector when missing.
std::vector<int> kv_i32_array(gguf_context * gctx, const char * key);

// Read a uint8 array (used for opaque-but-portable model metadata such as
// optional Tesseract DAWG components). Empty when missing or mistyped.
std::vector<uint8_t> kv_u8_array(gguf_context * gctx, const char * key);

// Read a float32 array (e.g. tokenizer.ggml.scores). Empty when missing/mistyped.
std::vector<float> kv_f32_array(gguf_context * gctx, const char * key);

// ---------------------------------------------------------------------------
// Pass 2: tensor allocation + weight data copy.
// ---------------------------------------------------------------------------

// CROSS-REPO TENSOR-MAP CONTRACT (read before changing the map type!)
// ------------------------------------------------------------------
// `core/gguf_loader.{h,cpp}` exists in BOTH CrispEmbed and CrispASR. When
// CrispEmbed builds, it pulls in CrispASR's `crisp_audio`/`crisp_lid` sources,
// and those compile against THIS header (crisp_audio links `crispembed-core`
// and uses its includes). CrispEmbed prefers `std::unordered_map` for tensor
// lookup (faster); CrispASR's standalone build prefers `std::map`.
//
// A consumer that does `ctx.tensors = std::move(wl.tensors)` therefore needs
// its own `tensors` field to be the SAME type as `WeightLoad::tensors` — but
// that type differs per repo. Hard-coding either type in the consumer breaks
// the other build, which is what caused the std::map<->unordered_map flip-flop
// war (CrispASR commits e6693b23/ad869798/d1bd3b91/1e4f1184/844f89d3, …).
//
// FIX: expose the type as a single alias `core_gguf::tensor_map`. Each repo's
// header defines it as its own choice; consumers (e.g. crisp_audio/audio_tower,
// crisp_lid/lid_cld3) declare their field as `core_gguf::tensor_map tensors;`
// so it AUTOMATICALLY matches whichever gguf_loader.h is compiled. Do not
// hard-code std::map / std::unordered_map in those consumer structs again.
using tensor_map = std::unordered_map<std::string, ggml_tensor *>;

struct WeightLoad {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    // Optional second backend buffer for tensors routed off-GPU. Non-null only
    // when load_weights_split() was used (CrispASR PLAN #69a pattern).
    ggml_backend_buffer_t buf_cpu = nullptr;
    // Overflow chunks from load_weights_split()'s chunked allocation (per-
    // allocation caps on some drivers); the first GPU/CPU buffer stays in
    // buf/buf_cpu, extra chunks live here for lifetime management only.
    std::vector<ggml_backend_buffer_t> split_bufs;
    tensor_map tensors;
    // Set only on the no-copy mmap path: the file stays mapped for the buffer's
    // lifetime (the buffer points directly at these pages). free_weights() unmaps.
    void * mmap_addr = nullptr;
    size_t mmap_len = 0;
    bool used_mmap = false; // true if the no-copy path was actually taken
};

// Load all tensor metadata + weights into a new ggml_context backed by
// a newly-allocated backend buffer. On success the WeightLoad struct is
// populated and the caller takes ownership of ctx/buf (typically moving
// them into the model struct).
//
// model_tag is used only in error messages ("parakeet: ...").
//
// try_mmap (opt-in, default off): when the backend device supports
// buffer_from_host_ptr (e.g. Metal/CPU unified memory), point the backend
// buffer directly at the mmap'd file instead of allocating a buffer and
// copying 2.x GB into it — halving resident memory and skipping the copy.
// Falls back to the copy path automatically if unsupported. Behaviour is
// otherwise identical (validated by tests/test_gguf_loader_mmap).
bool load_weights(const char * path, ggml_backend_t backend, const char * model_tag, WeightLoad & out,
                  bool try_mmap = false);

// Split-residency weight loader (logic synced from CrispASR PLAN #69a).
// Tensors for which `is_gpu(tensor_name, user) == true` are allocated on
// gpu_backend; the rest on cpu_backend. Compute graphs then follow weight
// residency. Uses the alloc+copy path (no persistent mmap — the split
// partition can't satisfy the contiguous-region requirement of the no-copy
// path). Caller owns out.ctx / out.buf (gpu partition) / out.buf_cpu (cpu
// partition) / out.split_bufs; free via free_weights(). Returns false on any
// allocation or read failure with partial state freed.
using IsGpuTensor = bool (*)(const char * tensor_name, void * user);
bool load_weights_split(const char * path, ggml_backend_t gpu_backend, ggml_backend_t cpu_backend, IsGpuTensor is_gpu,
                        void * user, const char * model_tag, WeightLoad & out);

// Free a backend buffer that came from one of this loader's weight-loading
// entry points, releasing the host mmap behind it when there is one.
//
// USE THIS INSTEAD OF ggml_backend_buffer_free() FOR ANY BUFFER OBTAINED FROM
// load_weights() / load_weights_split(), including after the buffer has been
// moved into a model struct. On the no-copy path
// (`load_weights(..., try_mmap=true)` on a device advertising
// buffer_from_host_ptr) the backend buffer is a view onto a host mapping the
// backend does not own — buffer_from_host_ptr has no deallocator parameter —
// so freeing the buffer alone leaves the weight file mapped for the life of
// the process. free_weights() reaches the same mapping through WeightLoad, but
// only for a caller that still holds the whole struct; a caller that moved
// `buf` into its model has this entry point and nothing else.
//
// Semantics:
//   * A null handle is a no-op, and the caller's handle is nulled on return,
//     so a second call cannot double-free or double-unmap.
//   * A buffer with no recorded mapping is the ordinary case — the copy path
//     maps nothing that outlives the load — and is released like any other
//     backend buffer.
//   * The backend buffer is freed before the region is unmapped, so a
//     device-side view of the pages never outlives them.
//
// This is also the entry point CrispASR's shared `crisp_audio` source calls,
// so the name and behaviour have to hold in both repos — see the cross-repo
// contract note above.
void release_weight_buffer(ggml_backend_buffer_t & buf);

// Free a WeightLoad's resources. Call when the model is being destroyed
// and the buffer/context are not held elsewhere. Releases every buffer
// through release_weight_buffer().
void free_weights(WeightLoad & wl);

// ---------------------------------------------------------------------------
// Tensor lookup helpers.
// ---------------------------------------------------------------------------

// Look up a tensor by name. Returns nullptr (silently) if missing.
// Uses `tensor_map` (see the cross-repo contract note above) so the signature
// tracks the per-repo map choice automatically.
ggml_tensor * try_get(const tensor_map & tensors, const char * name);

// Look up a tensor by name. Prints an error to stderr if missing but
// still returns nullptr — the caller decides whether a missing tensor
// is fatal.
ggml_tensor * require(const tensor_map & tensors, const char * name, const char * model_tag);

// Build a shell command that produces the formatted tensor name for a
// per-layer lookup. Avoids the snprintf(buf, sizeof(buf), "...", i) line
// that every loader repeats.
std::string format_layer_name(const char * fmt, int i);
std::string format_layer_name(const char * fmt, int i, int j);

} // namespace core_gguf
