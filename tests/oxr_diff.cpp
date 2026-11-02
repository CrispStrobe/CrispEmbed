// All-stage OXR comparison against an original pinned Python capture.
// Runs the actual image frontend and a composed prefill; quality/speed remain
// separate gates. Missing stages, shape changes and nonfinite values fail.
#include "oxr_test_common.h"
#include "crispembed_diff.h"
#include "core/clean_exit.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>

namespace {
struct record {
    crispembed_diff::Report diff;
    int visits = 0;
    bool finite = false, shape_equal = false;
    size_t count = 0;
    int argmax = -1;
};
struct state {
    crispembed_diff::Ref * reference;
    std::map<std::string, record> records;
};
std::vector<std::string> required() {
    std::vector<std::string> names = { "input.patches", "vision.patch_embed", "vision.position", "vision.embedding" };
    for (int i = 0; i < 27; ++i) names.push_back("vision.layer." + std::to_string(i));
    names.insert(names.end(), { "vision.merge", "visual.logits", "visual.probs", "visual.embeddings", "llm.input" });
    for (int i = 0; i < 28; ++i) names.push_back("llm.layer." + std::to_string(i));
    names.insert(names.end(), { "llm.norm", "llm.logits" });
    return names;
}
void observe(void * user, const char * name, const float * values, size_t count, const int64_t * shape, int ndim) {
    auto & capture = *static_cast<state *>(user);
    auto & rec = capture.records[name];
    ++rec.visits;
    rec.count = count;
    const auto expected = capture.reference->get_f32(name);
    auto expected_shape = capture.reference->shape(name);
    std::reverse(expected_shape.begin(), expected_shape.end());
    rec.shape_equal = shape && ndim > 0 && expected_shape == std::vector<int64_t>(shape, shape + ndim);
    rec.finite = values && count && expected.first && expected.second == count;
    if (rec.finite) {
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(values[i]) || !std::isfinite(expected.first[i])) {
                rec.finite = false;
                break;
            }
        }
    }
    if (rec.finite && rec.shape_equal) {
        rec.diff = capture.reference->compare(name, values, count);
        if (std::string(name) == "llm.logits") rec.argmax = int(std::max_element(values, values + count) - values);
    }
}
bool valid_report(const crispembed_diff::Report & diff) {
    return diff.found && std::isfinite(diff.max_abs) && std::isfinite(diff.mean_abs) && std::isfinite(diff.rms) &&
           std::isfinite(diff.cos_min) && std::isfinite(diff.mine_norm) && std::isfinite(diff.ref_norm);
}
bool exact_ids(const int32_t * ids, int count, const std::pair<const float *, size_t> & expected) {
    if (!ids || count < 0 || !expected.first || size_t(count) != expected.second) return false;
    for (int i = 0; i < count; ++i)
        if (float(ids[i]) != expected.first[i]) return false;
    return true;
}
int execute(int argc, char ** argv) {
    auto opt = oxr_test::parse(argc, argv);
    if (opt.reference.empty()) throw std::invalid_argument("--reference GGUF is required");
    crispembed_diff::Ref ref;
    if (!ref.load(opt.reference)) throw std::runtime_error("reference could not be loaded");
    if (ref.meta("oxr.model_revision") != "c708407f9d2b4df658f2543d9132b62bb7221df4" ||
        ref.meta("oxr.blueprint_revision") != "c1a1dcb312ee9f691404f127785510453c0d9350" ||
        ref.meta("oxr.model") != "SoMarkAI/OXR-1.0" || ref.meta("oxr.dtype") != "f32" ||
        ref.meta("oxr.prompt") != opt.prompt)
        throw std::invalid_argument("reference source, dtype or prompt does not match this protocol");
    auto * meta = core_gguf::open_metadata(opt.reference.c_str());
    if (!meta) throw std::runtime_error("reference metadata could not be loaded");
    const int generation_key = gguf_find_key(meta, "oxr.hf_generation");
    const bool profiles_match =
        core_gguf::kv_u32(meta, "oxr.min_pixels", 0) == uint32_t(opt.min_pixels) &&
        core_gguf::kv_u32(meta, "oxr.max_pixels", 0) == uint32_t(opt.max_pixels) && generation_key >= 0 &&
        gguf_get_kv_type(meta, generation_key) == GGUF_TYPE_BOOL &&
        gguf_get_val_bool(meta, generation_key) == (opt.hf_generation != 0) &&
        std::abs(core_gguf::kv_f32(meta, "oxr.repetition_penalty", -1.f) - opt.repetition_penalty) <= 1e-7f;
    core_gguf::free_metadata(meta);
    if (!profiles_match) throw std::invalid_argument("reference processor/generation profile differs");
    meta = core_gguf::open_metadata(opt.model.c_str());
    if (!meta) throw std::runtime_error("native metadata could not be loaded");
    const std::string recipe = core_gguf::kv_str(meta, "oxr.conversion.recipe", "");
    core_gguf::free_metadata(meta);
    if (recipe != "f32") throw std::invalid_argument("this primary stage protocol requires the F32 native recipe");
    const auto names = required();
    for (const auto & name : names)
        if (!ref.has(name)) throw std::invalid_argument("missing reference stage: " + name);
    const auto img = oxr_test::load_image(opt.image);
    // One generated token isolates prefill observations; ordinary completion
    // and repeated-context behavior are checked by the separate timing driver.
    opt.max_tokens = 1;
    auto ctx = oxr_test::load_model(opt);
    int plain_length = 0;
    const char * plain = oxr_ocr_recognize_raw(ctx.get(), img.rgb.data(), img.width, img.height, 3, &plain_length);
    if (!plain || !oxr_ocr_last_ok(ctx.get())) throw std::runtime_error(oxr_ocr_last_error(ctx.get()));
    const std::string plain_text(plain, size_t(plain_length));
    const std::string plain_raw = oxr_ocr_raw_text(ctx.get());
    int plain_count = 0;
    const auto * plain_output = oxr_ocr_completion_ids(ctx.get(), &plain_count);
    const std::vector<int32_t> plain_ids = plain_output && plain_count > 0
                                               ? std::vector<int32_t>(plain_output, plain_output + plain_count)
                                               : std::vector<int32_t>();
    const float plain_confidence = oxr_ocr_mean_confidence(ctx.get());
    const int plain_eos = oxr_ocr_stopped_eos(ctx.get()), plain_capped = oxr_ocr_capped(ctx.get());
    state capture{ &ref, {} };
    oxr_ocr_set_stage_observer(ctx.get(), observe, &capture);
    int length = 0;
    const char * text = oxr_ocr_recognize_raw(ctx.get(), img.rgb.data(), img.width, img.height, 3, &length);
    const bool ran = text && oxr_ocr_last_ok(ctx.get());
    int count = 0;
    const auto * ids = oxr_ocr_prompt_ids(ctx.get(), &count);
    const bool input_exact = exact_ids(ids, count, ref.get_f32("input_ids"));
    const bool grid_exact = exact_ids(oxr_ocr_grid_thw(ctx.get()), 3, ref.get_f32("grid_thw"));
    const auto expected_logits = ref.get_f32("llm.logits");
    const auto logits = capture.records.find("llm.logits");
    const int reference_argmax =
        expected_logits.first && expected_logits.second
            ? int(std::max_element(expected_logits.first, expected_logits.first + expected_logits.second) -
                  expected_logits.first)
            : -1;
    const bool argmax_equal =
        logits != capture.records.end() && reference_argmax >= 0 && logits->second.argmax == reference_argmax;
    const auto output = ref.get_f32("output_ids");
    const auto * completion = oxr_ocr_completion_ids(ctx.get(), &count);
    const bool first_token_exact =
        completion && count == 1 && output.first && output.second && float(completion[0]) == output.first[0];
    const bool observer_neutral =
        ran && plain_text == std::string(text, size_t(length)) && plain_raw == oxr_ocr_raw_text(ctx.get()) &&
        completion && count == plain_count && std::equal(plain_ids.begin(), plain_ids.end(), completion) &&
        std::isfinite(plain_confidence) && plain_confidence == oxr_ocr_mean_confidence(ctx.get()) &&
        plain_eos == oxr_ocr_stopped_eos(ctx.get()) && plain_capped == oxr_ocr_capped(ctx.get());
    bool pass = ran && input_exact && grid_exact && argmax_equal && first_token_exact && observer_neutral &&
                capture.records.size() == names.size();
    std::string first_failure;
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(12)
        << "{\"schema\":\"crispembed.oxr.stages.v1\",\"scope\":\"native image frontend and composed prefill\","
        << "\"conversion_recipe\":" << oxr_test::quote(recipe) << ",\"max_abs_budget\":" << opt.max_abs
        << ",\"min_cos_budget\":" << opt.min_cos << ",\"stages\":[";
    for (size_t i = 0; i < names.size(); ++i) {
        const auto found = capture.records.find(names[i]);
        const record absent;
        const auto & rec = found == capture.records.end() ? absent : found->second;
        const auto & diff = rec.diff;
        const double budget = names[i] == "input.patches" ? 2e-6 : opt.max_abs;
        const bool stage_pass = rec.visits == 1 && rec.finite && rec.shape_equal && valid_report(diff) &&
                                diff.max_abs <= budget && diff.cos_min >= opt.min_cos;
        if (!stage_pass) {
            pass = false;
            if (first_failure.empty()) first_failure = names[i];
        }
        out << (i ? "," : "") << "{\"name\":" << oxr_test::quote(names[i])
            << ",\"pass\":" << (stage_pass ? "true" : "false") << ",\"visits\":" << rec.visits
            << ",\"elements\":" << rec.count << ",\"shape_equal\":" << (rec.shape_equal ? "true" : "false")
            << ",\"finite\":" << (rec.finite ? "true" : "false");
        if (valid_report(diff))
            out << ",\"max_abs\":" << diff.max_abs << ",\"mean_abs\":" << diff.mean_abs << ",\"rms\":" << diff.rms
                << ",\"cos_min\":" << diff.cos_min << ",\"native_norm\":" << diff.mine_norm
                << ",\"reference_norm\":" << diff.ref_norm;
        out << '}';
    }
    out << "],\"input_ids_exact\":" << (input_exact ? "true" : "false")
        << ",\"grid_exact\":" << (grid_exact ? "true" : "false")
        << ",\"first_argmax_equal\":" << (argmax_equal ? "true" : "false")
        << ",\"first_token_exact\":" << (first_token_exact ? "true" : "false")
        << ",\"observer_neutrality_pass\":" << (observer_neutral ? "true" : "false")
        << ",\"first_failed_stage\":" << (first_failure.empty() ? "null" : oxr_test::quote(first_failure))
        << ",\"native_error\":" << oxr_test::quote(oxr_ocr_last_error(ctx.get()))
        << ",\"pass\":" << (pass ? "true" : "false") << ",\"decoded_quality_verified\":false,\"speed_verified\":false}";
    std::cout << out.str() << '\n';
    return pass ? 0 : 1;
}
} // namespace

int main(int argc, char ** argv) {
    int code = 2;
    try {
        code = execute(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "OXR stage comparator: %s\n", error.what());
    }
    core_util::clean_exit(code);
}
