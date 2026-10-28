// Authored weight-free Q4_K graph controls. Execute on hosted CI only.
#include "moedern_q4k_profile.h"
#include "moedern_family_engine.h"
#include "core/clean_exit.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
namespace {
void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
void bounds(const std::string & mutation) {
    // Authored storage-only fixture: one Q4_K row and31 F32 scalars. It is
    // deliberately not a model and must still fail the semantic metadata gate.
    std::vector<unsigned char> header, payload;
    auto integer = [&](uint64_t value, int width) {
        for (int i = 0; i < width; ++i) header.push_back(static_cast<unsigned char>(value >> (8 * i)));
    };
    auto text = [&](const std::string & value) {
        integer(value.size(), 8);
        header.insert(header.end(), value.begin(), value.end());
    };
    integer(0x46554747, 4);
    integer(3, 4);
    integer(32, 8);
    integer(0, 8);
    uint64_t offset = 0;
    for (int i = 0; i < 32; ++i) {
        text("owned." + std::to_string(mutation == "duplicate" && i == 1 ? 0 : i));
        integer(1, 4);
        integer(i ? 1 : mutation == "unaligned" ? 128 : 256, 8);
        integer(i                          ? GGML_TYPE_F32
                : mutation == "wrong-type" ? GGML_TYPE_Q4_0
                : mutation == "no-q4"      ? GGML_TYPE_F32
                                           : GGML_TYPE_Q4_K,
                4);
        integer(mutation == "overlap" && i == 1 ? 0 : offset, 8);
        const size_t bytes = i ? 4 : mutation == "no-q4" ? 1024 : 144;
        payload.resize(size_t(offset) + bytes, 0);
        offset = (offset + bytes + 31) / 32 * 32;
    }
    header.resize((header.size() + 31) / 32 * 32, 0);
    if (mutation == "gap") payload[144] = 1;
    header.insert(header.end(), payload.begin(), payload.end());
    if (mutation == "truncated") header.pop_back();
    if (mutation == "tail") header.push_back(1);
    const auto path =
        std::filesystem::temp_directory_path() /
        ("moedern-q4k-owned-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    struct cleanup {
        std::filesystem::path path;
        ~cleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } scratch{ path };
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(header.data()), header.size());
    }
    std::string error;
    const bool accepted = moedern_q4k::file_bounds(path.string().c_str(), error);
    require(accepted == (mutation == "valid"), "authored pre-allocation bounds decision differs");
    if (accepted) {
        gguf_init_params params{ true, nullptr };
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_from_file(path.string().c_str(), params),
                                                                 gguf_free);
        require(bool(meta) && !moedern_q4k::metadata(meta.get(), error),
                "storage fixture accepted as production metadata");
    }
}
void gates() {
    for (const char * name : { "CRISPEMBED_MOEDERN_COLBERT", "CRISPEMBED_MOEDERN_FAMILY", "CRISPEMBED_MOEDERN_QUANT",
                               "CRISPEMBED_MOEDERN_FAST" })
        setenv(name, "1", 1);
    const char * values[] = { nullptr, "", "0" };
    for (const char * value : values) {
        if (value)
            setenv("CRISPEMBED_MOEDERN_Q4_K_PADDED", value, 1);
        else
            unsetenv("CRISPEMBED_MOEDERN_Q4_K_PADDED");
        std::string failure;
        auto * ctx = moedern_family::init_q4k("unused.gguf", 2, failure);
        require(!ctx && failure.find("gates") != std::string::npos, "gate rejected after file access or treated0 asON");
    }
}
void unit(int logical, bool routed) {
    const int stored = int(moedern_q4k::storage_columns(logical)), outputs = 2, experts = routed ? 2 : 1, tokens = 2,
              top = 2;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(ggml_init({ 8 * 1024 * 1024, nullptr, false }),
                                                              ggml_free);
    require(bool(owner), "owned graph context");
    auto * g = owner.get();
    auto * w = ggml_new_tensor_3d(g, GGML_TYPE_Q4_K, stored, outputs, experts);
    std::vector<float> authored(size_t(stored) * outputs * experts), decoded(authored.size());
    for (int e = 0; e < experts; ++e)
        for (int o = 0; o < outputs; ++o)
            for (int c = 0; c < logical; ++c)
                authored[(size_t(e) * outputs + o) * stored + c] = float((e + 1) * (o ? -.5 : 1.));
    require(ggml_quantize_chunk(GGML_TYPE_Q4_K, authored.data(), w->data, 0, outputs * experts, stored, nullptr) ==
                ggml_nbytes(w),
            "actual Q4_K block encoding extent");
    ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(w->data, decoded.data(), int64_t(decoded.size()));
    auto * x = routed ? ggml_new_tensor_3d(g, GGML_TYPE_F32, logical, 1, tokens)
                      : ggml_new_tensor_2d(g, GGML_TYPE_F32, logical, tokens);
    for (int t = 0; t < tokens; ++t)
        for (int c = 0; c < logical; ++c) ggml_get_data_f32(x)[size_t(t) * logical + c] = float(t + 1);
    auto * padded = moedern_q4k::pad_input(g, x, logical, stored);
    require(padded->ne[0] == stored && padded->ne[1] == x->ne[1] && padded->ne[2] == x->ne[2],
            "padding must preserve token/expert axes");
    auto * chosen = ggml_new_tensor_2d(g, GGML_TYPE_I32, top, tokens);
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < top; ++k) static_cast<int32_t *>(chosen->data)[t * top + k] = (t + k) % experts;
    auto * y = routed ? ggml_mul_mat_id(g, w, padded, chosen) : ggml_mul_mat(g, w, padded);
    auto * graph = ggml_new_graph_custom(g, 128, false);
    ggml_build_forward_expand(graph, y);
    require(ggml_graph_compute_with_ctx(g, graph, 2) == GGML_STATUS_SUCCESS, "actual Q4_K graph compute");
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < (routed ? top : 1); ++k)
            for (int o = 0; o < outputs; ++o) {
                const int e = routed ? (t + k) % experts : 0;
                double expected = 0;
                for (int c = 0; c < logical; ++c)
                    expected += double(decoded[(size_t(e) * outputs + o) * stored + c]) * float(t + 1);
                const auto index = o + outputs * (k + (routed ? top : 1) * t);
                require(std::isfinite(ggml_get_data_f32(y)[index]) &&
                            std::abs(double(ggml_get_data_f32(y)[index]) - expected) <= .002,
                        "Q4_K semantic matmul reference differs");
            }
    if (stored > logical) {
        const auto * data = ggml_get_data_f32(padded);
        for (int t = 0; t < tokens; ++t)
            for (int c = logical; c < stored; ++c)
                require(data[size_t(t) * stored + c] == 0.f, "padded activation columns must be exactly zero");
    }
    bool rejected = false;
    try {
        (void)moedern_q4k::pad_input(g, x, logical, stored + 256);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "excess padding accepted");
}
int run() {
    require(GGML_FTYPE_MOSTLY_Q4_K == 12, "pinned actual Q4_K file-type ABI");
    require(ggml_blck_size(GGML_TYPE_Q4_K) == 256 && ggml_type_size(GGML_TYPE_Q4_K) == 144, "pinned actual Q4_K ABI");
    require(!moedern_q4k::compresses(16) && moedern_q4k::compresses(128), "narrow precision policy");
    for (int width : { 128, 640, 1024 })
        for (bool routed : { false, true }) unit(width, routed);
    for (const char * mutation :
         { "valid", "duplicate", "unaligned", "wrong-type", "no-q4", "overlap", "truncated", "tail", "gap" })
        bounds(mutation);
    gates();
    std::cout << "PASS6 authored actual Q4_K matmul/id controls, zero column padding and excess-padding rejection\n";
    std::cout << "PASS9 authored storage bounds/semantic rejection controls and3 gate-before-file controls\n";
    return 0;
}
} // namespace
int main() {
    try {
        core_util::clean_exit(run());
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        core_util::clean_exit(1);
    }
}
