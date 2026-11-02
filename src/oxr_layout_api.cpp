// OXR-1.0 D-FINE layout C API: a thin JSON view over layout_detect (dfine).
// The JSON body is the OXR layout service's /inference response
// (services/layout_analyze/dfine_la/confidence.py format_predictions, as
// serialized by Starlette's compact JSONResponse) for one page, so it can be
// passed unchanged to OXR's parse_layout_batch. See licenses/oxr-LICENSE.
#include "crispembed.h"
#include "core/env_gate.h"
#include "core/python_float_repr.h"
#include "layout_detect.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

struct crispembed_oxr_layout_context {
    layout_detect::context * model = nullptr;
    std::string json, error;
    ~crispembed_oxr_layout_context() { layout_detect::free(model); }
};

using core_pyrepr::append_python_float;

extern "C" {

crispembed_oxr_layout_context * crispembed_oxr_layout_init(const char * model_path, int n_threads) {
    if (!model_path || !*model_path || !core_env::on("CRISPEMBED_OXR") ||
        !core_env::on("CRISPEMBED_ACCEPT_OXR_LICENSE"))
        return nullptr;
    try {
        auto * ctx = new crispembed_oxr_layout_context();
        if (!layout_detect::load(&ctx->model, model_path, n_threads < 1 ? 1 : n_threads) ||
            layout_detect::arch(ctx->model) != layout_detect::arch_id::dfine) {
            delete ctx;
            return nullptr;
        }
        return ctx;
    } catch (const std::exception &) {
        return nullptr;
    }
}

void crispembed_oxr_layout_free(crispembed_oxr_layout_context * ctx) {
    delete ctx;
}

const char * crispembed_oxr_layout_detect(crispembed_oxr_layout_context * ctx, const uint8_t * pixels, int width,
                                          int height, int channels, float confidence) {
    if (!ctx) return nullptr;
    ctx->json.clear();
    ctx->error.clear();
    try {
        const bool policy = std::isfinite(confidence) && confidence >= 0;
        if (std::isfinite(confidence) && confidence > 1) throw std::invalid_argument("confidence must be in [0, 1]");
        std::vector<layout_detect::candidate> cands;
        if (!layout_detect::detect_candidates(ctx->model, pixels, width, height, channels, cands))
            throw std::runtime_error(layout_detect::last_error(ctx->model));
        std::string labels = "[[", boxes = "[[", scores = "[[";
        bool first = true;
        for (const auto & c : cands) {
            // format_predictions: keep = score >= threshold (only with a policy).
            if (policy && !((double)c.score >= (double)confidence)) continue;
            if (!first) {
                labels += ',';
                boxes += ',';
                scores += ',';
            }
            first = false;
            labels += std::to_string(c.raw_class);
            boxes += '[';
            for (int j = 0; j < 4; j++) {
                if (j) boxes += ',';
                append_python_float(boxes, (double)c.box[j]);
            }
            boxes += ']';
            append_python_float(scores, (double)c.score);
        }
        labels += "]]";
        boxes += "]]";
        scores += "]]";
        std::string params = "{}";
        if (policy) {
            params = "{\"confidence\":";
            append_python_float(params, (double)confidence);
            params += ",\"class_confidences\":{}}";
        }
        ctx->json = "{\"status\":\"success\",\"params\":" + params + ",\"data\":{\"labels\":" + labels +
                    ",\"boxes\":" + boxes + ",\"scores\":" + scores + "}}";
        return ctx->json.c_str();
    } catch (const std::exception & e) {
        ctx->error = e.what();
        return nullptr;
    }
}

const char * crispembed_oxr_layout_timings(crispembed_oxr_layout_context * ctx) {
    if (!ctx) return nullptr;
    const auto t = layout_detect::last_timings(ctx->model);
    std::string out = "{\"preprocess_ms\":";
    append_python_float(out, t.preprocess_ms);
    out += ",\"encoder_ms\":";
    append_python_float(out, t.encoder_ms);
    out += ",\"decoder_ms\":";
    append_python_float(out, t.decoder_ms);
    out += ",\"postprocess_ms\":";
    append_python_float(out, t.postprocess_ms);
    out += ",\"total_ms\":";
    append_python_float(out, t.total_ms);
    out += '}';
    ctx->json = out;
    return ctx->json.c_str();
}

const char * crispembed_oxr_layout_last_error(const crispembed_oxr_layout_context * ctx) {
    if (!ctx)
        return "null OXR layout context (init requires CRISPEMBED_OXR=1, CRISPEMBED_ACCEPT_OXR_LICENSE=1 "
               "and an OXR D-FINE GGUF)";
    return ctx->error.c_str();
}

} // extern "C"
