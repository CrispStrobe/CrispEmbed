#include "lighton4b_native_context.h"
#include "lighton4b_bf16_dequant_weights.h"
#include "core/clean_exit.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
using namespace qwen35_ocr;
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
static float rne(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
    std::memcpy(&x, &bits, 4);
    return x;
}
static int run(int argc, char ** argv) {
    if (argc != 2) return 2;
    int checks = 0;
    auto check = [&](bool ok) {
        if (!ok) {
            std::fprintf(stderr, "q6 control %d failed\n", checks);
            std::exit(1);
        }
        ++checks;
    };
    for (auto * name : { "CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER", "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT",
                         "CRISPEMBED_LIGHTON4B_QUANT", "CRISPEMBED_QWEN35_QUANT", "CRISPEMBED_QWEN35_OCR",
                         "CRISPEMBED_QWEN35_STREAM_PROJECTION" })
        gate(name, "1");
    auto * header = ggml_init({ 426 * ggml_tensor_overhead() + 4096, nullptr, true });
    check(header != nullptr);
    core_gguf::tensor_map tensors;
    std::ifstream input(argv[1]);
    std::string name;
    int kind;
    int64_t ne[4];
    while (input >> name >> kind >> ne[0] >> ne[1] >> ne[2] >> ne[3]) {
        auto * t = ggml_new_tensor(header, ggml_type(kind), 4, ne);
        check(t && !t->data && tensors.emplace(name, t).second);
    }
    check(input.eof() && tensors.size() == 426);
    lighton4b_text_owner owner;
    auto * meta = gguf_init_empty();
    gguf_set_val_str(meta, "qwen35_ocr.mixed_quant_recipe", lighton4b_q6_recipe);
    gguf_set_val_u32(meta, "general.file_type", 15);
    for (auto * value : { static_cast<const char *>(nullptr), "", "0" }) {
        gate("CRISPEMBED_LIGHTON4B_Q6_K", value);
        check(!lighton4b_q6_metadata(meta));
        check(!map_lighton4b_text(tensors, owner));
    }
    gate("CRISPEMBED_LIGHTON4B_Q6_K", "1");
    check(lighton4b_q6_metadata(meta));
    check(map_lighton4b_text(tensors, owner));
    check(lighton4b_quant_recipe_matches(owner.decoder, lighton4b_quant_recipe::q4km_q6_q8head));
    check(!lighton4b_quant_recipe_matches(owner.decoder, lighton4b_quant_recipe::q4_k));
    check(owner.decoder.embedding == owner.decoder.head && owner.decoder.head->type == GGML_TYPE_Q8_0);
    // Every selected or unselected quantized matrix is independently corrupted.
    for (auto & row : tensors)
        if (row.second->type != GGML_TYPE_F32) {
            const auto saved = row.second->type;
            row.second->type = saved == GGML_TYPE_Q6_K ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
            check(!lighton4b_quant_recipe_matches(owner.decoder, lighton4b_quant_recipe::q4km_q6_q8head));
            row.second->type = saved;
        }
    gguf_set_val_u32(meta, "general.file_type", 14);
    check(!lighton4b_q6_metadata(meta));
    gguf_set_val_u32(meta, "general.file_type", 18);
    check(!lighton4b_q6_metadata(meta));
    gguf_set_val_str(meta, "general.file_type", "15");
    check(!lighton4b_q6_metadata(meta));
    gguf_set_val_u32(meta, "general.file_type", 15);
    gguf_set_val_str(meta, "qwen35_ocr.mixed_quant_recipe", "q4_k_m");
    check(!lighton4b_q6_metadata(meta));
    gguf_free(meta);
    ggml_free(header);
    auto * ctx = ggml_init({ 65536, nullptr, true });
    auto * backend = ggml_backend_cpu_init();
    auto * q6 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 256, 257);
    auto * f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 257);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr && ggml_type_size(GGML_TYPE_Q6_K) == 210);
    std::vector<uint8_t> storage(ggml_nbytes(q6));
    std::vector<float> raw(256 * 257), rounded(raw.size()), got(raw.size()), hidden(256 * 3);
    const uint16_t bits[] = { 0x3800, 0xb400, 0x0000 };
    const float scales[] = { 0.5f, -0.25f, 0.0f };
    for (size_t block = 0; block < 257; ++block) {
        auto * bytes = storage.data() + 210 * block;
        for (int j = 0; j < 208; ++j) bytes[j] = uint8_t((block * 37 + size_t(j) * 13 + 17) % 256);
        std::memcpy(bytes + 208, &bits[block % 3], 2);
        // Independent output-index scalar decoder, not ggml's to_float loop.
        for (int j = 0; j < 256; ++j) {
            const int half = j / 128, p = j % 128;
            const int lo = (bytes[half * 64 + p % 64] >> (p >= 64 ? 4 : 0)) & 15;
            const int hi = (bytes[128 + half * 32 + p % 32] >> (2 * (p / 32))) & 3;
            int scale = bytes[192 + j / 16];
            if (scale >= 128) scale -= 256;
            raw[block * 256 + j] = scales[block % 3] * scale * ((lo | (hi << 4)) - 32);
        }
    }
    ggml_backend_tensor_set(q6, storage.data(), 0, storage.size());
    ggml_set_name(q6, "l.layers.0.mlp.down_proj.weight");
    for (auto * value : { static_cast<const char *>(nullptr), "0", "1" }) {
        gate("CRISPEMBED_LIGHTON4B_Q6_K", value);
        gate("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", "0");
        for (int threads : { 1, 4 })
            check(tensor_floats(q6, got.data(), threads) && !std::memcmp(got.data(), raw.data(), raw.size() * 4));
    }
    gate("CRISPEMBED_LIGHTON4B_Q6_K", "1");
    gate("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", "1");
    for (size_t i = 0; i < raw.size(); ++i) rounded[i] = rne(raw[i]);
    check(std::memcmp(rounded.data(), raw.data(), raw.size() * 4) != 0);
    check(tensor_floats(q6, got.data(), 4) && !std::memcmp(got.data(), rounded.data(), raw.size() * 4));
    ggml_backend_tensor_set(f32, rounded.data(), 0, rounded.size() * 4);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = float(int(i % 37) - 18) / 32;
    std::vector<std::vector<float>> actual, expected;
    check(project_cpu({ q6 }, hidden.data(), 256, 3, 2, actual) &&
          project_cpu({ f32 }, hidden.data(), 256, 3, 2, expected));
    check(actual[0] == expected[0]);
    std::vector<float> rows, ref;
    check(project_rows_cpu(q6, hidden.data(), 256, 3, 2, 31, rows) &&
          project_rows_cpu(f32, hidden.data(), 256, 3, 2, 31, ref) && rows == ref);
    std::vector<uint8_t> after(storage.size());
    ggml_backend_tensor_get(q6, after.data(), 0, after.size());
    check(after == storage);
    gate("CRISPEMBED_LIGHTON4B_Q6_K", "0");
    check(tensor_floats(q6, got.data(), 1) && got == raw);
    check(!project_rows_cpu(q6, hidden.data(), 256, 3, 2, 31, rows) && rows.empty());
    // Actual grouped-GDN Q6 projection, prefill and immutable-cache decode.
    auto * grouped = ggml_init({ 65536, nullptr, true });
    linear_weights w;
    w.qkv = ggml_new_tensor_2d(grouped, GGML_TYPE_Q6_K, 256, 32);
    auto * qkv_ref = ggml_new_tensor_2d(grouped, GGML_TYPE_F32, 256, 32);
    w.z = ggml_new_tensor_2d(grouped, GGML_TYPE_F32, 256, 16);
    w.a = ggml_new_tensor_2d(grouped, GGML_TYPE_F32, 256, 2);
    w.b = ggml_new_tensor_2d(grouped, GGML_TYPE_F32, 256, 2);
    w.out = ggml_new_tensor_2d(grouped, GGML_TYPE_F32, 16, 256);
    w.conv = ggml_new_tensor_3d(grouped, GGML_TYPE_F32, 4, 1, 32);
    w.a_log = ggml_new_tensor_1d(grouped, GGML_TYPE_F32, 2);
    w.dt_bias = ggml_new_tensor_1d(grouped, GGML_TYPE_F32, 2);
    w.norm = ggml_new_tensor_1d(grouped, GGML_TYPE_F32, 8);
    auto * grouped_buffer = ggml_backend_alloc_ctx_tensors(grouped, backend);
    check(grouped_buffer != nullptr);
    ggml_backend_tensor_set(w.qkv, storage.data(), 0, 32 * 210);
    ggml_backend_tensor_set(qkv_ref, raw.data(), 0, 32 * 256 * 4);
    for (auto * t : { w.z, w.a, w.b, w.out, w.conv, w.a_log, w.dt_bias, w.norm }) {
        std::vector<float> values(size_t(ggml_nelements(t)));
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = t == w.norm ? 1.0f : t == w.a_log ? -1.0f : float(int(i % 7) - 3) / 1024;
        ggml_backend_tensor_set(t, values.data(), 0, values.size() * 4);
    }
    auto reference = w;
    reference.qkv = qkv_ref;
    std::vector<float> small(512);
    for (size_t i = 0; i < small.size(); ++i) small[i] = hidden[i] / 65536;
    gate("CRISPEMBED_QWEN35_GDN_GROUPED_HEADS", "1");
    gate("CRISPEMBED_LIGHTON4B_Q6_K", "1");
    gate("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", "0");
    projected_linear_outputs actual_core, expected_core;
    check(linear_grouped_cpu(w, 1, small.data(), 1, 2, 2, nullptr, nullptr, actual_core) &&
          linear_grouped_cpu(reference, 1, small.data(), 1, 2, 2, nullptr, nullptr, expected_core));
    check(actual_core.qkv == expected_core.qkv && actual_core.attention == expected_core.attention &&
          actual_core.core.state == expected_core.core.state &&
          actual_core.core.conv_history == expected_core.core.conv_history);
    const auto saved_state = actual_core.core.state, saved_conv = actual_core.core.conv_history;
    projected_linear_outputs decode, decode_ref;
    check(linear_grouped_cpu(w, 1, small.data() + 256, 1, 1, 2, saved_conv.data(), saved_state.data(), decode) &&
          linear_grouped_cpu(reference, 1, small.data() + 256, 1, 1, 2, expected_core.core.conv_history.data(),
                             expected_core.core.state.data(), decode_ref));
    check(decode.attention == decode_ref.attention && decode.core.state == decode_ref.core.state &&
          actual_core.core.state == saved_state && actual_core.core.conv_history == saved_conv);
    gate("CRISPEMBED_LIGHTON4B_Q6_K", "0");
    check(!linear_grouped_cpu(w, 1, small.data(), 1, 2, 2, nullptr, nullptr, decode) && decode.qkv.empty() &&
          decode.core.state.empty());
    ggml_backend_buffer_free(grouped_buffer);
    ggml_free(grouped);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::printf("Q6_POLICY passed=1 checks=%d q6_matrices=32 q8_tied_head=1 independent_dequant=1 "
                "synthetic_values=65792 grouped_q6_prefill_decode=1 storage_unchanged=1 model_payload=0\n",
                checks);
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
