// crispembed-spacy-diff: stage-by-stage native vs original-spaCy comparison.
//
// Runs every corpus text through the public counted-text C ABI with a trace
// callback, and compares each traced stage against the crispembed-diff
// reference written by tools/spacy/dump_spacy_reference.py (original thinc
// modules, optionally with injected GGUF weights). Prints, per stage in
// pipeline order, the worst per-row cosine (cos_min), the worst whole-tensor
// cosine (cos_global) and max |diff| over all cases, then the first stage
// whose cos_min falls below the threshold (CRISPEMBED_DIFF_COS_THRESHOLD,
// default 0.999) -- where a port defect would live. Engine and precision are
// chosen by the usual environment (CRISPEMBED_SPACY_GGML, _QUANT, _THREADS).
//
//   crispembed-spacy-diff --ref ref.gguf --corpus corpus.json --bundle DIR [--model M.gguf]
#include "crispembed_diff.h"
#include "core/json_value.h"
#include "core/sha256.h"
#include "spacy_pipeline.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Worst {
    size_t seen = 0, missing = 0;
    float cos_min = 1.0f, cos_global = 1.0f, max_abs = 0.0f;
};

struct Capture {
    const crispembed_diff::Ref * ref;
    int case_index;
    std::vector<std::string> * order;
    std::map<std::string, Worst> * worst;
};

int on_stage(void * user, const char * stage, const float * values, const size_t * shape, size_t rank) {
    auto * capture = static_cast<Capture *>(user);
    const std::string name(stage);
    if (name.rfind("parser.step.", 0) == 0) return 0; // per-step states have no module reference
    char key[96];
    std::snprintf(key, sizeof(key), "c%03d.%s", capture->case_index, stage);
    size_t count = 1;
    for (size_t i = 0; i < rank; ++i) count *= shape[i];
    auto & worst = (*capture->worst)[name];
    if (worst.seen == 0 && worst.missing == 0) capture->order->push_back(name);
    if (!capture->ref->has(key)) {
        ++worst.missing;
        return 0;
    }
    const auto report = capture->ref->compare(key, values, count);
    if (!report.found || report.n_elem != count) {
        std::fprintf(stderr, "%s: element count %zu differs from reference\n", key, count);
        worst.cos_min = worst.cos_global = -1.0f;
        ++worst.seen;
        return 0;
    }
    ++worst.seen;
    worst.cos_min = std::min(worst.cos_min, report.cos_min);
    worst.cos_global = std::min(worst.cos_global, report.cos_global);
    worst.max_abs = std::max(worst.max_abs, report.max_abs);
    return 0;
}

std::string read_file(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    std::stringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

} // namespace

int main(int argc, char ** argv) {
    std::string ref_path, corpus_path, bundle, model;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--ref") ref_path = argv[i + 1];
        else if (flag == "--corpus") corpus_path = argv[i + 1];
        else if (flag == "--bundle") bundle = argv[i + 1];
        else if (flag == "--model") model = argv[i + 1];
    }
    if (ref_path.empty() || corpus_path.empty() || bundle.empty()) {
        std::fprintf(stderr, "usage: %s --ref ref.gguf --corpus corpus.json --bundle DIR [--model M.gguf]\n", argv[0]);
        return 2;
    }
    if (model.empty()) model = bundle + "/model.gguf";
    const std::string resources = bundle + "/resources.json";
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) {
        std::fprintf(stderr, "cannot load reference %s\n", ref_path.c_str());
        return 2;
    }
    const auto corpus = core_json_value::parse(read_file(corpus_path));
    const auto * texts = corpus.get("texts");
    if (!texts || texts->kind != core_json_value::value::array_t) {
        std::fprintf(stderr, "corpus has no texts array\n");
        return 2;
    }
    char error[1024] = { 0 };
    const std::string model_sha = core_sha256::file(model), resources_sha = core_sha256::file(resources);
    auto * context = crispembed_spacy_create_with_resources(model.c_str(), model_sha.c_str(), resources.c_str(),
                                                            resources_sha.c_str(), error, sizeof(error));
    if (!context) {
        std::fprintf(stderr, "native admission failed: %s\n", error);
        return 2;
    }
    std::vector<std::string> order;
    std::map<std::string, Worst> worst;
    int failures = 0;
    for (size_t i = 0; i < texts->a.size(); ++i) {
        const auto & text = texts->a[i].s;
        Capture capture{ &ref, static_cast<int>(i), &order, &worst };
        crispembed_spacy_document * document = nullptr;
        const int status = crispembed_spacy_analyze_utf8(context, text.data(), text.size(), &document, on_stage,
                                                         &capture, nullptr, nullptr, error, sizeof(error));
        if (status != CRISPEMBED_SPACY_OK) {
            std::fprintf(stderr, "case %zu: analyze failed (%d): %s\n", i, status, error);
            ++failures;
        }
        crispembed_spacy_document_free(document);
    }
    crispembed_spacy_free(context);

    // Same override as crispembed_diff::Report (CRISPEMBED_DIFF_COS_THRESHOLD).
    float threshold = 0.999f;
    if (const char * value = std::getenv("CRISPEMBED_DIFF_COS_THRESHOLD")) {
        char * end = nullptr;
        const float parsed = std::strtof(value, &end);
        if (end && end != value && *end == '\0' && parsed > -1.0f && parsed <= 1.0f) threshold = parsed;
    }
    std::printf("%-36s %6s %10s %10s %10s\n", "stage", "cases", "cos_min", "cos_global", "max_abs");
    std::string first_divergent;
    for (const auto & name : order) {
        const auto & w = worst[name];
        if (w.seen == 0) {
            std::printf("%-36s %6s %10s %10s %10s  (no reference)\n", name.c_str(), "-", "-", "-", "-");
            continue;
        }
        const bool pass = w.cos_min >= threshold;
        if (!pass && first_divergent.empty()) first_divergent = name;
        std::printf("%-36s %6zu %10.6f %10.6f %10.3e  %s\n", name.c_str(), w.seen, w.cos_min, w.cos_global, w.max_abs,
                    pass ? "PASS" : "FAIL");
    }
    std::printf("first divergent stage (cos_min < %.6f): %s\n", threshold,
                first_divergent.empty() ? "none" : first_divergent.c_str());
    return failures || !first_divergent.empty() ? 1 : 0;
}
