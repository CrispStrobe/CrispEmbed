// MinerU original-module stage comparison. This is a test executable only.
// Exact grid and prompt IDs come from the frozen Python GGUF, never factorized
// from token counts. Every expected stage must be observed and satisfy BOTH
// absolute error and per-row cosine budgets; missing/nonfinite data fails.
//
// test-mineru25-diff model.gguf original.gguf [threads=4] [max_abs=.002] [min_cos=.99999]
// Requires CRISPEMBED_MINERU25_PRO=1. This runs the reference-patch-fed composed
// graph. Pair it with test-mineru25-run for native preprocessing, decoded tokens,
// EOS, and warm speed; a passing stage receipt alone is not OCR qualification.

#include "qwen2vl_ocr.h"
#include "crispembed_diff.h"
#include "core/clean_exit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {
// Per-row detail: a worst-row cosine alone cannot tell a near-zero-norm row from a
// normal row computed wrongly.
struct row_stats {
    bool valid = false;
    size_t width = 0, rows = 0, worst = 0, below_099 = 0;
    double worst_cos = 1, worst_ref_norm = 0, worst_native_norm = 0, median_ref_norm = 0, p01_cos = 1, p50_cos = 1;
};

size_t row_width(const std::string & name) {
    if (name == "input_patches") return 1176;
    if (name == "vis_patch_embed" || name.rfind("vis_layer_", 0) == 0) return 1280;
    if (name == "vis_merger_output" || name == "llm_embed" || name == "llm_final_norm" ||
        name.rfind("llm_layer_", 0) == 0)
        return 896;
    return 0;
}

row_stats rows_of(const std::string & name, const float * mine, const float * ref, size_t count) {
    row_stats r;
    r.width = row_width(name);
    if (!r.width || count % r.width) return r;
    r.rows = count / r.width;
    std::vector<double> cosines(r.rows), ref_norms(r.rows);
    for (size_t row = 0; row < r.rows; ++row) {
        double dot = 0, a = 0, b = 0;
        for (size_t j = row * r.width; j < (row + 1) * r.width; ++j) {
            dot += double(mine[j]) * ref[j];
            a += double(mine[j]) * mine[j];
            b += double(ref[j]) * ref[j];
        }
        const double c = a == 0 && b == 0 ? 1.0 : a == 0 || b == 0 ? 0.0 : dot / std::sqrt(a * b);
        cosines[row] = c;
        ref_norms[row] = std::sqrt(b);
        if (c < 0.99) ++r.below_099;
        if (row == 0 || c < r.worst_cos) {
            r.worst_cos = c;
            r.worst = row;
            r.worst_ref_norm = std::sqrt(b);
            r.worst_native_norm = std::sqrt(a);
        }
    }
    auto pick = [](std::vector<double> v, double q) {
        std::nth_element(v.begin(), v.begin() + size_t(q * (v.size() - 1)), v.end());
        return v[size_t(q * (v.size() - 1))];
    };
    r.median_ref_norm = pick(ref_norms, 0.5);
    r.p01_cos = pick(cosines, 0.01);
    r.p50_cos = pick(cosines, 0.5);
    r.valid = true;
    return r;
}

struct stage_record {
    row_stats rows;
    crispembed_diff::Report result;
    bool finite = false;
    bool same_size = false;
    size_t count = 0;
    int visits = 0;
};

struct observation {
    crispembed_diff::Ref * reference = nullptr;
    std::map<std::string, stage_record> stages;
    std::vector<std::string> order;
};

void observe(void * user, const char * name, const float * data, size_t count) {
    auto & state = *static_cast<observation *>(user);
    auto & record = state.stages[name];
    if (!record.visits) state.order.emplace_back(name);
    ++record.visits;
    record.count = count;
    const auto expected = state.reference->get_f32(name);
    record.same_size = expected.first && expected.second == count;
    record.finite = data && count > 0;
    if (record.finite) {
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(data[i]) || (record.same_size && !std::isfinite(expected.first[i]))) {
                record.finite = false;
                break;
            }
        }
    }
    if (record.finite && record.same_size) {
        record.result = state.reference->compare(name, data, count);
        record.rows = rows_of(name, data, expected.first, count);
    }
}

std::vector<std::string> required_stages() {
    std::vector<std::string> result = { "input_patches", "vis_patch_embed" };
    for (int i = 0; i < 32; ++i) result.push_back("vis_layer_" + std::to_string(i));
    result.insert(result.end(), { "vis_merger_output", "input_ids", "mrope_positions", "llm_embed" });
    for (int i = 0; i < 24; ++i) result.push_back("llm_layer_" + std::to_string(i));
    result.insert(result.end(), { "llm_final_norm", "llm_logits_last" });
    return result;
}

bool finite_report(const crispembed_diff::Report & result) {
    return result.found && std::isfinite(result.max_abs) && std::isfinite(result.mean_abs) &&
           std::isfinite(result.cos_min) && std::isfinite(result.rms) && std::isfinite(result.mine_norm) &&
           std::isfinite(result.ref_norm);
}

int execute(int argc, char ** argv) {
    if (argc < 3 || argc > 6) {
        std::fprintf(stderr, "Usage: %s model.gguf reference.gguf [threads=4] [max_abs=.002] [min_cos=.99999]\n",
                     argv[0]);
        return 2;
    }
    const int threads = argc > 3 ? std::stoi(argv[3]) : 4;
    const double max_abs = argc > 4 ? std::stod(argv[4]) : .002;
    const double min_cos = argc > 5 ? std::stod(argv[5]) : .99999;
    if (threads <= 0 || !std::isfinite(max_abs) || max_abs < 0 || !std::isfinite(min_cos) || min_cos < 0 ||
        min_cos > 1) {
        std::fprintf(stderr, "Invalid thread count or numerical budget\n");
        return 2;
    }
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    if (ref.meta("mineru25.model") != "opendatalab/MinerU2.5-Pro-2604-1.2B" ||
        ref.meta("mineru25.blueprint_revision") != "c9797cde8f54cc48306196fc6aebe73ea31dd4a3") {
        std::fprintf(stderr, "Reference source identity does not match this protocol\n");
        return 2;
    }
    const auto expected = required_stages();
    for (const auto & name : expected) {
        if (!ref.has(name)) {
            std::fprintf(stderr, "Required reference stage missing: %s\n", name.c_str());
            return 2;
        }
    }
    const auto grid_values = ref.get_f32("grid_thw");
    if (!grid_values.first || grid_values.second != 3) {
        std::fprintf(stderr, "Reference must contain exact grid_thw tensor\n");
        return 2;
    }
    int32_t grid[3] = {};
    for (int i = 0; i < 3; ++i) {
        const float value = grid_values.first[i];
        if (!std::isfinite(value) || value < 1 || value > 8192 || value != std::floor(value)) return 2;
        grid[i] = static_cast<int32_t>(value);
    }
    const int64_t patch_count = int64_t(grid[0]) * grid[1] * grid[2];
    if (grid[0] != 1 || grid[1] % 2 || grid[2] % 2 || patch_count > 8192) return 2;
    const auto patches = ref.get_f32("input_patches");
    if (!patches.first || patches.second != size_t(patch_count) * 1176) return 2;
    const auto input_ids = ref.get_f32("input_ids");
    if (!input_ids.first || !input_ids.second || input_ids.second >= 8192) return 2;
    std::vector<int32_t> ids(input_ids.second);
    for (size_t i = 0; i < ids.size(); ++i) {
        const float value = input_ids.first[i];
        if (!std::isfinite(value) || value < 0 || value >= 151936 || value != std::floor(value)) return 2;
        ids[i] = static_cast<int32_t>(value);
    }
    if (std::count(ids.begin(), ids.end(), 151655) != patch_count / 4) {
        std::fprintf(stderr, "Prompt image placeholder count does not match exact grid\n");
        return 2;
    }

    qwen2vl_ocr::context ctx;
    if (!qwen2vl_ocr::load(ctx, argv[1], threads, 0)) return 2;
    if (!ctx.is_mineru25_pro) {
        std::fprintf(stderr, "This test requires the opted-in MinerU profile\n");
        qwen2vl_ocr::free_(ctx);
        return 2;
    }
    observation observations;
    observations.reference = &ref;
    ctx.diff_ref_path = argv[2];
    ctx.stage_observer = observe;
    ctx.stage_observer_user = &observations;
    qwen2vl_ocr::vision_result vision;
    if (!qwen2vl_ocr::encode_vision(ctx, patches.first, int(patch_count), grid, vision)) {
        qwen2vl_ocr::free_(ctx);
        return 2;
    }
    qwen2vl_ocr::image_input image;
    image.image_embeds = vision.image_embeds;
    image.n_image_tokens = vision.n_merged;
    image.grid_thw = grid;
    image.n_images = 1;
    qwen2vl_ocr::llm_result llm;
    const bool ran = qwen2vl_ocr::run_llm_forward(ctx, ids.data(), int(ids.size()), llm, &image, true, true);
    bool argmax_equal = false;
    int native_argmax = -1, reference_argmax = -1;
    const auto expected_logits = ref.get_f32("llm_logits_last");
    if (ran && llm.logits && llm.n_logits == 1 && expected_logits.second == size_t(llm.vocab_size)) {
        native_argmax = int(std::max_element(llm.logits, llm.logits + llm.vocab_size) - llm.logits);
        reference_argmax = int(std::max_element(expected_logits.first, expected_logits.first + expected_logits.second) -
                               expected_logits.first);
        argmax_equal = native_argmax == reference_argmax;
    }
    if (llm.logits) std::free(llm.logits);
    if (llm.hidden) std::free(llm.hidden);
    if (llm.kv_graph_ctx) ggml_free(llm.kv_graph_ctx);
    qwen2vl_ocr::vision_result_free(vision);
    qwen2vl_ocr::free_(ctx);

    bool all_pass = ran && argmax_equal;
    std::string first_failure;
    // v2 (reported beside, never replacing, v1): v1's fixed 0.002 absolute budget is
    // below one float32 ulp for Qwen2 massive activations (~5e4). v2 scales by the
    // stage's reference RMS: cos_min >= min_cos, relative RMS <= 1e-5, and
    // max_abs <= max(0.002, 0.005 * reference RMS). Chosen after inspecting v1
    // failures (hosted 41214087617); v1 pass and the exit code are unchanged.
    bool all_pass_v2 = ran && argmax_equal;
    std::string first_failure_v2;
    std::printf("{\"schema\":\"crispembed.mineru25.stages.v1\",\"scope\":\"reference-patch-fed composed graph\","
                "\"max_abs_budget\":%.17g,\"min_cos_budget\":%.17g,\"grid_thw\":[%d,%d,%d],\"stages\":[",
                max_abs, min_cos, grid[0], grid[1], grid[2]);
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto & name = expected[i];
        const auto found = observations.stages.find(name);
        const bool exists = found != observations.stages.end();
        const stage_record empty;
        const auto & record = exists ? found->second : empty;
        const auto & r = record.result;
        const bool exact_stage = name == "input_ids" || name == "mrope_positions";
        const bool pass = exists && record.visits == 1 && record.finite && record.same_size && finite_report(r) &&
                          r.max_abs <= (exact_stage ? 0.0 : max_abs) && r.cos_min >= min_cos;
        if (!pass) {
            all_pass = false;
            if (first_failure.empty()) first_failure = name;
        }
        const double ref_rms = record.count ? double(r.ref_norm) / std::sqrt(double(record.count)) : 0.0;
        const double rel_rms = ref_rms > 0 ? double(r.rms) / ref_rms : (r.rms == 0 ? 0.0 : INFINITY);
        const bool pass_v2 =
            exists && record.visits == 1 && record.finite && record.same_size && finite_report(r) &&
            (exact_stage ? r.max_abs == 0.0 : r.max_abs <= std::max(max_abs, 0.005 * ref_rms) && rel_rms <= 1e-5) &&
            r.cos_min >= min_cos;
        if (!pass_v2) {
            all_pass_v2 = false;
            if (first_failure_v2.empty()) first_failure_v2 = name;
        }
        std::printf("%s{\"name\":\"%s\",\"pass\":%s,\"pass_v2\":%s,\"observed\":%s,\"visits\":%d,\"elements\":%zu,"
                    "\"finite\":%s",
                    i ? "," : "", name.c_str(), pass ? "true" : "false", pass_v2 ? "true" : "false",
                    exists ? "true" : "false", record.visits, record.count, record.finite ? "true" : "false");
        if (finite_report(r)) {
            std::printf(",\"max_abs\":%.9g,\"mean_abs\":%.9g,\"rms\":%.9g,\"rel_rms\":%.9g,\"cos_min\":%.9g,"
                        "\"reference_norm\":%.9g,\"native_norm\":%.9g",
                        r.max_abs, r.mean_abs, r.rms, rel_rms, r.cos_min, r.ref_norm, r.mine_norm);
            std::printf(",\"cos_mean\":%.9g", r.cos_mean);
        }
        if (record.rows.valid) {
            const auto & w = record.rows;
            std::printf(",\"rows\":{\"width\":%zu,\"count\":%zu,\"below_0_99\":%zu,\"p01_cos\":%.9g,\"p50_cos\":%.9g,"
                        "\"worst_row\":%zu,\"worst_cos\":%.9g,\"worst_ref_norm\":%.9g,\"worst_native_norm\":%.9g,"
                        "\"median_ref_norm\":%.9g}",
                        w.width, w.rows, w.below_099, w.p01_cos, w.p50_cos, w.worst, w.worst_cos, w.worst_ref_norm,
                        w.worst_native_norm, w.median_ref_norm);
        }
        std::printf("}");
    }
    std::printf("],\"native_first_argmax\":%d,\"reference_first_argmax\":%d,\"first_argmax_equal\":%s,"
                "\"first_failed_stage\":%s,\"pass\":%s,\"first_failed_stage_v2\":%s,\"pass_v2\":%s,"
                "\"decoded_quality_verified\":false,\"speed_verified\":false}\n",
                native_argmax, reference_argmax, argmax_equal ? "true" : "false",
                first_failure.empty() ? "null" : ("\"" + first_failure + "\"").c_str(), all_pass ? "true" : "false",
                first_failure_v2.empty() ? "null" : ("\"" + first_failure_v2 + "\"").c_str(),
                all_pass_v2 ? "true" : "false");
    return all_pass ? 0 : 1;
}
} // namespace

int main(int argc, char ** argv) {
    int code = 2;
    try {
        code = execute(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "MinerU stage test: %s\n", error.what());
    }
    core_util::clean_exit(code);
}
