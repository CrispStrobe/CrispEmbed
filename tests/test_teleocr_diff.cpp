// Full native TeleOCR pipeline versus observed, pinned Transformers outputs.
// Test-only observer code is compiled separately from the production archive.
#include "qwen2vl_ocr.h"
#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "core/json.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

extern "C" unsigned char * stbi_load(const char *, int *, int *, int *, int);
extern "C" void stbi_image_free(void *);

namespace {

struct Stage {
    std::string name;
    size_t count = 0;
    int64_t width = 0;
    double max_abs = 0.0, mean_abs = 0.0, rms_error = 0.0;
    double ref_rms = 0.0, native_rms = 0.0, min_cos = 1.0;
    bool pass = false;
    int argmax_native = -1, argmax_ref = -1;
};

struct Reference {
    gguf_context * meta = nullptr;
    ggml_context * tensors = nullptr;
    std::ifstream file;
    std::map<std::string, Stage> stages;
    std::string metadata;
    std::string error;
    std::vector<int> generated;

    ~Reference() {
        if (meta) gguf_free(meta);
        if (tensors) ggml_free(tensors);
    }

    bool load(const char * path) {
        gguf_init_params params = { true, &tensors };
        meta = gguf_init_from_file(path, params);
        file.open(path, std::ios::binary);
        if (!meta || !file) return false;
        const int64_t key = gguf_find_key(meta, "teleocr.reference.metadata");
        if (key < 0 || gguf_get_kv_type(meta, key) != GGUF_TYPE_STRING) return false;
        metadata = gguf_get_val_str(meta, key);
        std::vector<std::string> required = { "input_patches",     "image_grid",     "prompt_ids",
                                              "mrope_positions",   "generated_ids",  "vis_patch_embed",
                                              "vis_pre_merger",    "merger_output",  "llm_embed",
                                              "llm_L0_Q_pre_rope", "llm_L0_Q_rope",  "llm_L0_attn_out",
                                              "llm_L0_post_attn",  "llm_final_norm", "logits_step0" };
        for (int i = 0; i < 32; ++i) required.push_back("vis_layer_" + std::to_string(i));
        for (int i = 0; i < 28; ++i) required.push_back("llm_layer_" + std::to_string(i));
        if ((size_t)gguf_get_n_tensors(meta) != required.size()) return false;
        for (const auto & name : required)
            if (gguf_find_tensor(meta, name.c_str()) < 0) return false;
        return true;
    }

    bool read(const char * name, std::vector<float> & values) {
        const int64_t index = gguf_find_tensor(meta, name);
        ggml_tensor * tensor = ggml_get_tensor(tensors, name);
        if (index < 0 || !tensor || tensor->type != GGML_TYPE_F32) return false;
        values.resize((size_t)ggml_nelements(tensor));
        const uint64_t offset = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, index);
        file.clear();
        file.seekg((std::streamoff)offset);
        file.read(reinterpret_cast<char *>(values.data()), (std::streamsize)(values.size() * sizeof(float)));
        return bool(file);
    }

    static void observe(void * opaque, const char * name, const float * data, size_t n, int64_t width) {
        auto & self = *static_cast<Reference *>(opaque);
        // Prefill stages are intentionally captured once; repeated decoding has
        // different shapes and is checked by the full sequence and EOS result.
        if (self.stages.count(name)) return;
        // Retain the actual stop/tokens even when a candidate emits a different
        // number of IDs; a length mismatch must not erase its native evidence.
        if (std::string(name) == "generated_ids") {
            for (size_t i = 0; i < n; ++i) self.generated.push_back((int)data[i]);
        }
        if (const char * dump_dir = std::getenv("CRISPEMBED_TELEOCR_DUMP_STAGES")) {
            const std::string stage_name(name);
            if (*dump_dir && (stage_name == "vis_layer_16" || stage_name == "vis_layer_17" ||
                              stage_name == "llm_layer_1" || stage_name == "llm_layer_2")) {
                std::ofstream dump(std::string(dump_dir) + "/" + stage_name + ".f32", std::ios::binary);
                dump.write(reinterpret_cast<const char *>(data), (std::streamsize)(n * sizeof(float)));
                if (!dump) self.error = "Cannot write optional native stage: " + stage_name;
            }
        }
        std::vector<float> expected;
        if (!self.read(name, expected)) {
            self.error = std::string("Missing reference tensor: ") + name;
            return;
        }
        Stage s;
        s.name = name;
        s.count = n;
        s.width = width;
        ggml_tensor * expected_tensor = ggml_get_tensor(self.tensors, name);
        if (n != expected.size() || width <= 0 || n % (size_t)width != 0 || !expected_tensor ||
            expected_tensor->ne[0] != width) {
            self.error = std::string("Shape mismatch: ") + name;
            self.stages.emplace(name, s);
            return;
        }
        bool finite = true;
        double diff_square = 0.0, ref_square = 0.0, native_square = 0.0;
        for (size_t row = 0; row < n; row += (size_t)width) {
            double aa = 0.0, bb = 0.0, ab = 0.0;
            for (size_t j = row; j < row + (size_t)width; ++j) {
                const double a = data[j], b = expected[j];
                finite = finite && std::isfinite(a) && std::isfinite(b);
                const double d = std::abs(a - b);
                s.max_abs = std::max(s.max_abs, d);
                s.mean_abs += d;
                diff_square += d * d;
                aa += a * a;
                bb += b * b;
                ab += a * b;
            }
            const double cosine = aa == 0.0 && bb == 0.0 ? 1.0 : aa == 0.0 || bb == 0.0 ? 0.0 : ab / std::sqrt(aa * bb);
            s.min_cos = std::min(s.min_cos, cosine);
            native_square += aa;
            ref_square += bb;
        }
        if (n) {
            s.mean_abs /= n;
            s.rms_error = std::sqrt(diff_square / n);
            s.ref_rms = std::sqrt(ref_square / n);
            s.native_rms = std::sqrt(native_square / n);
        }
        const bool exact = s.name == "image_grid" || s.name == "prompt_ids" || s.name == "mrope_positions" ||
                           s.name == "generated_ids";
        const double tolerance = exact ? 0.0 : s.name == "input_patches" ? 2e-6 : 0.002;
        s.pass = finite && s.max_abs <= tolerance && (exact || s.min_cos >= 0.99999);
        if (s.name == "logits_step0") {
            s.argmax_native = (int)(std::max_element(data, data + n) - data);
            s.argmax_ref = (int)(std::max_element(expected.begin(), expected.end()) - expected.begin());
            s.pass = s.pass && s.argmax_native == s.argmax_ref;
        }
        std::fprintf(stderr, "%s: max_abs=%.9g cos_min=%.12g rms=%.9g/%g %s\n", name, s.max_abs, s.min_cos,
                     s.native_rms, s.ref_rms, s.pass ? "PASS" : "FAIL");
        self.stages.emplace(name, s);
    }
};

static std::string json_string(const std::string & value) {
    return "\"" + core_json::json_escape(value) + "\"";
}

int run(int argc, char ** argv) {
    if (argc < 5 || argc > 7) {
        std::fprintf(stderr,
                     "Usage: %s MODEL.gguf REFERENCE.gguf IMAGE.png REPORT.json [threads=2] [max_tokens=1024]\n",
                     argv[0]);
        return 2;
    }
    const int threads = argc > 5 ? std::atoi(argv[5]) : 2;
    const int cap = argc > 6 ? std::atoi(argv[6]) : 1024;
    if (threads < 1 || cap < 1) return 2;
    Reference reference;
    if (!reference.load(argv[2])) {
        std::fprintf(stderr, "Cannot open TeleOCR reference\n");
        return 2;
    }
    std::string prompt, expected_text;
    size_t end = 0;
    const size_t prompt_at = core_json::json_find_key_value(reference.metadata, "prompt");
    const size_t text_at = core_json::json_find_key_value(reference.metadata, "text");
    if (prompt_at == std::string::npos || text_at == std::string::npos ||
        !core_json::json_decode_string(reference.metadata, prompt_at, prompt, end) ||
        !core_json::json_decode_string(reference.metadata, text_at, expected_text, end))
        return 2;
    if (prompt.empty()) return 2;
    int width = 0, height = 0, channels = 0;
    unsigned char * pixels = stbi_load(argv[3], &width, &height, &channels, 3);
    if (!pixels) return 2;
    qwen2vl_ocr_context * ctx = qwen2vl_ocr_init(argv[1], threads);
    if (!ctx) {
        stbi_image_free(pixels);
        return 2;
    }
    qwen2vl_ocr_test_set_observer(ctx, Reference::observe, &reference);
    qwen2vl_ocr_set_max_tokens(ctx, cap);
    if (!qwen2vl_ocr_set_prompt(ctx, prompt.c_str())) {
        qwen2vl_ocr_free(ctx);
        stbi_image_free(pixels);
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    int text_length = 0;
    const char * result = qwen2vl_ocr_recognize_raw(ctx, pixels, width, height, 3, &text_length);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::string text = result ? std::string(result, (size_t)text_length) : "";
    const bool result_ok = result != nullptr;
    bool confidence_pass = true;
    int confidence_count = 0;
    const float * confidence = qwen2vl_ocr_confidences(ctx, &confidence_count);
    for (int i = 0; i < confidence_count; ++i) {
        confidence_pass = confidence_pass && confidence && std::isfinite(confidence[i]) && confidence[i] >= 0.0f &&
                          confidence[i] <= 1.0f;
    }
    confidence_pass = confidence_pass && confidence_count > 0 && (size_t)confidence_count == reference.generated.size();
    qwen2vl_ocr_test_set_observer(ctx, nullptr, nullptr);
    qwen2vl_ocr_free(ctx);
    stbi_image_free(pixels);
    bool complete = true;
    for (int64_t i = 0; i < gguf_get_n_tensors(reference.meta); ++i) {
        const char * name = gguf_get_tensor_name(reference.meta, i);
        if (!reference.stages.count(name)) {
            std::fprintf(stderr, "Unobserved required stage: %s\n", name);
            complete = false;
        }
    }
    bool numerical_pass = complete && reference.error.empty();
    for (const auto & item : reference.stages) numerical_pass = numerical_pass && item.second.pass;
    const bool eos =
        !reference.generated.empty() && (reference.generated.back() == 151645 || reference.generated.back() == 151643);
    const auto ids = reference.stages.find("generated_ids");
    const bool decoded_pass =
        result_ok && text == expected_text && ids != reference.stages.end() && ids->second.pass && eos;
    const bool pass = numerical_pass && decoded_pass && confidence_pass;
    std::ofstream report(argv[4]);
    if (!report) return 2;
    report << std::setprecision(12) << "{\n  \"pass\": " << (pass ? "true" : "false")
           << ",\n  \"numerical_pass\": " << (numerical_pass ? "true" : "false")
           << ",\n  \"decoded_pass\": " << (decoded_pass ? "true" : "false")
           << ",\n  \"coverage_complete\": " << (complete ? "true" : "false")
           << ",\n  \"confidence_pass\": " << (confidence_pass ? "true" : "false")
           << ",\n  \"reached_eos\": " << (eos ? "true" : "false") << ",\n  \"seconds\": " << elapsed
           << ",\n  \"threads\": " << threads << ",\n  \"tokens\": " << reference.generated.size()
           << ",\n  \"confidence_count\": " << confidence_count << ",\n  \"text\": " << json_string(text)
           << ",\n  \"error\": " << json_string(reference.error) << ",\n  \"generated_ids\": [";
    for (size_t i = 0; i < reference.generated.size(); ++i) report << (i ? "," : "") << reference.generated[i];
    report << "],\n  \"stages\": [\n";
    size_t count = 0;
    for (const auto & item : reference.stages) {
        const Stage & s = item.second;
        report << (count++ ? ",\n" : "") << "    {\"name\": " << json_string(s.name) << ", \"elements\": " << s.count
               << ", \"row_width\": " << s.width << ", \"max_abs\": " << s.max_abs << ", \"mean_abs\": " << s.mean_abs
               << ", \"rms_error\": " << s.rms_error << ", \"native_rms\": " << s.native_rms
               << ", \"ref_rms\": " << s.ref_rms << ", \"min_cos\": " << s.min_cos
               << ", \"argmax_native\": " << s.argmax_native << ", \"argmax_ref\": " << s.argmax_ref
               << ", \"pass\": " << (s.pass ? "true" : "false") << "}";
    }
    report << "\n  ]\n}\n";
    std::fprintf(stderr, "TeleOCR: stages=%zu numerical=%d decoded=%d EOS=%d seconds=%.3f\n", reference.stages.size(),
                 numerical_pass, decoded_pass, eos, elapsed);
    std::printf("%s\n", text.c_str());
    return pass ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
