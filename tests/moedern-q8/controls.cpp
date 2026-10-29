// Authored weight-free profile/storage/actual Q8 graph controls. Hosted execution only.
#include "moedern_q8_profile.h"
#include "moedern_family_engine.h"
#include "moedern_text_frontend.h"
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
void metadata(const moedern_q8::pin & pin) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(ggml_init({ 1024 * 1024, nullptr, true }), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_empty(), gguf_free);
    require(tensors && g, "owned metadata contexts");
    require(!moedern_q8::storage_metadata(g.get()), "missing Q8 storage metadata accepted");
    for (const auto & row : std::vector<std::pair<const char *, const char *>>{
             { "general.architecture", "modernbert_hash_moe_v10" },
             { "general.license", "apache-2.0" },
             { "modernbert_hash_moe_v10.source_revision", pin.revision },
             { "moedern.packed.format", moedern_q8::FORMAT },
             { "modernbert_hash_moe_v10.quant_recipe", moedern_q8::RECIPE },
             { "moedern.packed.model_sha256", pin.model_sha256 },
             { "moedern.packed.original_sha256", pin.original_report_sha256 },
             { "moedern.packed.tokenizer_architecture", "moedern_unigram_tokenizer" },
             { "moedern.packed.tokenizer_sha256", "3ce7c3fa8b10249c5b704026566d183adb3de7b1879e7174e4654cdfd54ac4b1" },
             { "moedern.packed.variant", pin.variant },
             { "moedern.packed.composed_source", "1a053116908224cf1c9358d7d24c85ba36e82eff" },
             { "moedern.packed.composed_report_sha256",
               "98af43c6780bbd11b1ec9ec4f26c7b2e3a2348507072408408bf726b2e89482e" } })
        gguf_set_val_str(g.get(), row.first, row.second);
    gguf_set_val_u32(g.get(), "general.file_type", core_gguf_file_type::q8_0);
    gguf_set_val_u32(g.get(), "general.quantization_version", GGML_QNT_VERSION);
    for (uint64_t i = 0; i < pin.tensors; ++i) {
        auto * t = ggml_new_tensor_1d(tensors.get(),
                                      i < moedern_q8::expected_quantized(pin) ? GGML_TYPE_Q8_0 : GGML_TYPE_F32, 32);
        ggml_set_name(t, ("owned." + std::to_string(i)).c_str());
        gguf_add_tensor(g.get(), t);
    }
    std::string error;
    require(moedern_q8::metadata(g.get(), error) == &pin, "authored typed profile metadata rejected");
    for (uint32_t wrong : { 0u, 2u, 12u, 14u, 18u }) {
        gguf_set_val_u32(g.get(), "general.file_type", wrong);
        require(!moedern_q8::metadata(g.get(), error), "non-Q8 header accepted");
    }
    gguf_set_val_str(g.get(), "general.file_type", "7");
    require(!moedern_q8::metadata(g.get(), error), "string Q8 header accepted");
    gguf_set_val_u32(g.get(), "general.file_type", core_gguf_file_type::q8_0);
    gguf_set_val_u32(g.get(), "general.quantization_version", GGML_QNT_VERSION + 1);
    require(!moedern_q8::metadata(g.get(), error), "future quantization version accepted");
    gguf_set_val_u32(g.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_str(g.get(), "moedern.packed.format", "crispembed-moedern-family-f32-v1");
    require(!moedern_q8::metadata(g.get(), error), "old F32 profile silently upgraded to Q8");
    gguf_set_val_str(g.get(), "moedern.packed.format", moedern_q8::FORMAT);
    gguf_set_val_str(g.get(), "moedern.packed.model_sha256", "wrong");
    require(!moedern_q8::metadata(g.get(), error), "wrong original model identity accepted");
}
void bounds(const std::string & mutation) {
    // Structural inventory only; owned names must never qualify as real model.
    std::vector<unsigned char> header, payload;
    auto integer = [&](uint64_t v, unsigned bytes) {
        for (unsigned i = 0; i < bytes; ++i) header.push_back(static_cast<unsigned char>(v >> (8 * i)));
    };
    auto text = [&](const std::string & v) {
        integer(v.size(), 8);
        header.insert(header.end(), v.begin(), v.end());
    };
    integer(0x46554747, 4);
    integer(3, 4);
    integer(32, 8);
    integer(0, 8);
    uint64_t offset = 0;
    for (int i = 0; i < 32; ++i) {
        text("owned." + std::to_string(mutation == "duplicate" && i == 1 ? 0 : i));
        integer(1, 4);
        const bool quantized = i < 20;
        const uint64_t columns = quantized ? (mutation == "unaligned" && i == 0 ? 16 : 32)
                                 : i == 20 ? 1484448 - 20 * 32 - 11 + (mutation == "parameters")
                                           : 1;
        integer(columns, 8);
        const auto type =
            quantized ? (mutation == "wrong-type" && i == 0 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0) : GGML_TYPE_F32;
        integer(type, 4);
        integer(mutation == "overlap" && i == 1 ? 0 : offset, 8);
        const uint64_t bytes = quantized ? 34 : columns * 4;
        payload.resize(size_t(offset + bytes), 0);
        offset = (offset + bytes + 31) / 32 * 32;
    }
    header.resize((header.size() + 31) / 32 * 32, 0);
    if (mutation == "gap") payload[34] = 1;
    header.insert(header.end(), payload.begin(), payload.end());
    if (mutation == "truncated") header.pop_back();
    if (mutation == "tail") header.push_back(1);
    const auto path =
        std::filesystem::temp_directory_path() /
        ("moedern-q8-owned-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    struct cleanup {
        std::filesystem::path p;
        ~cleanup() {
            std::error_code ec;
            std::filesystem::remove(p, ec);
        }
    } owned{ path };
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(header.data()), header.size());
    }
    std::string error;
    require(moedern_q8::file_bounds(path.string().c_str(), error) == (mutation == "valid"),
            "authored Q8 pre-allocation bounds differ");
}
void gates() {
    for (const char * key :
         { "CRISPEMBED_MOEDERN_COLBERT", "CRISPEMBED_MOEDERN_FAMILY", "CRISPEMBED_MOEDERN_QUANT",
           "CRISPEMBED_MOEDERN_FAST", "CRISPEMBED_MOEDERN_TEXT_FRONTEND", "CRISPEMBED_MOEDERN_PACKED" })
        setenv(key, "1", 1);
    unsetenv("CRISPEMBED_MOEDERN_Q4_K_PADDED");
    for (const char * gate : { "CRISPEMBED_MOEDERN_Q8_0", "CRISPEMBED_MOEDERN_QUANT", "CRISPEMBED_MOEDERN_FAST",
                               "CRISPEMBED_MOEDERN_FAMILY" }) {
        for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
            for (const char * key : { "CRISPEMBED_MOEDERN_Q8_0", "CRISPEMBED_MOEDERN_QUANT", "CRISPEMBED_MOEDERN_FAST",
                                      "CRISPEMBED_MOEDERN_FAMILY" })
                setenv(key, "1", 1);
            if (value)
                setenv(gate, value, 1);
            else
                unsetenv(gate);
            std::string error;
            require(!moedern_family::init_q8("unused.gguf", 2, error) && error.find("gates") != std::string::npos,
                    "Q8 dependency gateOFF/0 reached file access");
            moedern_colbert::text_frontend front;
            require(!front.load_packed_q8("unused.gguf", error) && error.find("gates") != std::string::npos &&
                        front.family_revision().empty(),
                    "Q8 frontend gate/recovery state differs");
        }
    }
    for (const char * key : { "CRISPEMBED_MOEDERN_Q8_0", "CRISPEMBED_MOEDERN_QUANT", "CRISPEMBED_MOEDERN_FAST",
                              "CRISPEMBED_MOEDERN_FAMILY" })
        setenv(key, "1", 1);
    setenv("CRISPEMBED_MOEDERN_Q8_0", "1", 1);
    setenv("CRISPEMBED_MOEDERN_Q4_K_PADDED", "1", 1);
    std::string error;
    require(!moedern_family::init_q8("unused.gguf", 2, error) && error.find("Q4_K_PADDED OFF") != std::string::npos,
            "concurrent Q4/Q8 gates accepted");
    unsetenv("CRISPEMBED_MOEDERN_Q4_K_PADDED");
}
void graph(int width, bool routed) {
    const int outputs = 2, experts = routed ? 2 : 1, tokens = 2, top = 2;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(ggml_init({ 8 * 1024 * 1024, nullptr, false }),
                                                              ggml_free);
    require(bool(owner), "owned Q8 graph context");
    auto * g = owner.get();
    auto * w = ggml_new_tensor_3d(g, GGML_TYPE_Q8_0, width, outputs, experts);
    std::vector<float> authored(size_t(width) * outputs * experts), decoded(authored.size());
    for (int e = 0; e < experts; ++e)
        for (int o = 0; o < outputs; ++o)
            for (int c = 0; c < width; ++c)
                authored[(size_t(e) * outputs + o) * width + c] = float((e + 1) * (o ? -63.5 : 127.));
    require(ggml_quantize_chunk(GGML_TYPE_Q8_0, authored.data(), w->data, 0, outputs * experts, width, nullptr) ==
                ggml_nbytes(w),
            "actual Q8 block extent differs");
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(w->data, decoded.data(), decoded.size());
    auto * x = routed ? ggml_new_tensor_3d(g, GGML_TYPE_F32, width, 1, tokens)
                      : ggml_new_tensor_2d(g, GGML_TYPE_F32, width, tokens);
    for (int t = 0; t < tokens; ++t)
        for (int c = 0; c < width; ++c) ggml_get_data_f32(x)[size_t(t) * width + c] = float((t + 1) * 127);
    auto * chosen = ggml_new_tensor_2d(g, GGML_TYPE_I32, top, tokens);
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < top; ++k) static_cast<int32_t *>(chosen->data)[t * top + k] = (t + k) % experts;
    auto * y = routed ? ggml_mul_mat_id(g, w, x, chosen) : ggml_mul_mat(g, w, x);
    auto * cg = ggml_new_graph_custom(g, 128, false);
    ggml_build_forward_expand(cg, y);
    require(ggml_graph_compute_with_ctx(g, cg, 2) == GGML_STATUS_SUCCESS, "actual Q8 graph failed");
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < (routed ? top : 1); ++k)
            for (int o = 0; o < outputs; ++o) {
                const int e = routed ? (t + k) % experts : 0;
                double expected = 0;
                for (int c = 0; c < width; ++c)
                    expected += double(decoded[(size_t(e) * outputs + o) * width + c]) * float((t + 1) * 127);
                const auto index = o + outputs * (k + (routed ? top : 1) * t);
                require(std::isfinite(ggml_get_data_f32(y)[index]) &&
                            std::abs(double(ggml_get_data_f32(y)[index]) - expected) <= .002,
                        "Q8 same-weight authored dot differs");
            }
}
int run() {
    require(ggml_blck_size(GGML_TYPE_Q8_0) == 32 && ggml_type_size(GGML_TYPE_Q8_0) == 34, "actual Q8 ABI differs");
    for (const auto & pin : moedern_q8::pins()) metadata(pin);
    for (const char * mutation :
         { "valid", "duplicate", "unaligned", "wrong-type", "parameters", "overlap", "gap", "truncated", "tail" })
        bounds(mutation);
    gates();
    for (int width : { 128, 640, 1024 })
        for (bool routed : { false, true }) graph(width, routed);
    std::cout << "PASS3 named Q8 metadata profiles,9 storage mutations,gateOFF/0/concurrent checks,6 actual Q8 "
                 "direct/expert graphs\n";
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
