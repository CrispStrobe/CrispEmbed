// OXR-1.0 native document route public API (CRISPEMBED_OXR_NATIVE_DOC=1).
// See src/oxr_document.h, licenses/oxr-LICENSE and licenses/oxr-NOTICE.
#include "core/env_gate.h"
#include "core/json.h"
#include "marker_document.h"
#include "oxr_api_internal.h"

#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string quote(const std::string & value) {
    return "\"" + core_json::json_escape_strict(value) + "\"";
}
std::vector<uint8_t> to_rgb(const uint8_t * pixels, int width, int height, int channels) {
    if (!pixels || width <= 0 || height <= 0 || uint64_t(width) * uint64_t(height) > 100000000ULL ||
        (channels != 1 && channels != 3 && channels != 4))
        throw std::invalid_argument("OXR pages must be gray/RGB/RGBA uint8 with valid dimensions <=100 megapixels");
    std::vector<uint8_t> rgb(size_t(width) * size_t(height) * 3);
    for (size_t i = 0; i < size_t(width) * size_t(height); ++i)
        for (int c = 0; c < 3; ++c)
            rgb[3 * i + size_t(c)] = pixels[i * size_t(channels) + size_t(channels == 1 ? 0 : c)];
    return rgb;
}

std::string trace_json(const std::vector<oxr_document::trace_entry> & trace) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << '[';
    for (size_t i = 0; i < trace.size(); ++i) {
        const auto & t = trace[i];
        if (i) out << ',';
        out << "{\"kind\":" << quote(t.kind) << ",\"page\":" << t.page << ",\"block\":" << t.block
            << ",\"prompt\":" << quote(t.prompt) << ",\"width\":" << t.width << ",\"height\":" << t.height
            << ",\"rgb_sha256\":" << quote(t.rgb_sha256) << ",\"max_tokens\":" << t.max_tokens
            << ",\"repetition_penalty\":" << t.repetition_penalty << ",\"finish_reason\":" << quote(t.finish_reason)
            << ",\"result\":" << (t.native_json.empty() ? "null" : t.native_json) << '}';
    }
    out << ']';
    return out.str();
}
} // namespace

extern "C" {
const char * crispembed_oxr_process_pages(crispembed_oxr_context * ctx, const uint8_t * const * pixels,
                                          const int * widths, const int * heights, const int * channels, int n_pages,
                                          const crispembed_oxr_layout_region * regions, const int * region_counts,
                                          const char * layout_json, const char * options_json) {
    if (!ctx) return nullptr;
    ctx->error.clear();
    ctx->document_json.clear();
    ctx->trace_json = "[]";
    ctx->has_document = false;
    try {
        if (!core_env::on("CRISPEMBED_OXR_NATIVE_DOC"))
            throw std::runtime_error("the native OXR document route requires CRISPEMBED_OXR_NATIVE_DOC=1");
        if (!core_env::on("CRISPEMBED_OXR")) throw std::runtime_error("CRISPEMBED_OXR=1 is required");
        if (n_pages <= 0 || n_pages > 10000 || !pixels || !widths || !heights || !channels)
            throw std::invalid_argument("OXR requires at least one page with pixels and dimensions");
        if ((regions || region_counts) && layout_json)
            throw std::invalid_argument("Pass regions or layout_json, not both");
        if (!layout_json && !region_counts)
            throw std::invalid_argument("Document processing requires supplied layout regions or layout_json");
        std::vector<std::vector<uint8_t>> storage;
        std::vector<oxr_document::rgb_page> pages;
        for (int p = 0; p < n_pages; ++p) storage.push_back(to_rgb(pixels[p], widths[p], heights[p], channels[p]));
        for (int p = 0; p < n_pages; ++p) pages.push_back({ widths[p], heights[p], storage[size_t(p)].data() });
        core_json_value::value layout;
        if (layout_json) {
            layout = core_json_value::parse(layout_json);
        } else {
            std::vector<oxr_document::page_layout> supplied(static_cast<size_t>(n_pages));
            size_t at = 0;
            for (int p = 0; p < n_pages; ++p) {
                if (region_counts[p] < 0 || (region_counts[p] && !regions))
                    throw std::invalid_argument("invalid region count");
                for (int k = 0; k < region_counts[p]; ++k, ++at) {
                    const auto & r = regions[at];
                    supplied[size_t(p)].push_back({ r.label, r.x1, r.y1, r.x2, r.y2, r.score });
                }
            }
            layout = oxr_document::layout_response(supplied);
        }
        const oxr_document::options opts = oxr_document::parse_options(options_json ? options_json : "");
        oxr_document::recognizer rec = [ctx](const oxr_document::rec_request & r) {
            oxr_document::rec_result out;
            out.native_json = oxr_predict_json(ctx, r.rgb, r.width, r.height, 3, r.prompt.c_str(), r.max_tokens,
                                               float(r.repetition_penalty), out.eos, out.capped, out.text);
            out.finish_reason = out.eos ? "stop" : out.capped ? "length" : "unknown";
            return out;
        };
        std::vector<oxr_document::trace_entry> trace;
        try {
            ctx->document = oxr_document::process_pages(pages, layout, opts, rec, &trace);
        } catch (...) {
            ctx->trace_json = trace_json(trace);
            throw;
        }
        ctx->trace_json = trace_json(trace);
        ctx->document_json = ctx->document.json;
        ctx->has_document = true;
        return ctx->document_json.c_str();
    } catch (const std::exception & error) {
        ctx->error = error.what();
        return nullptr;
    }
}

const char * crispembed_oxr_last_trace(const crispembed_oxr_context * ctx) {
    return ctx ? ctx->trace_json.c_str() : "[]";
}

const char * crispembed_oxr_render_document(crispembed_oxr_context * ctx, const char * format) {
    if (!ctx) return nullptr;
    ctx->error.clear();
    try {
        if (!ctx->has_document) throw std::runtime_error("no native OXR document has been processed");
        if (!format) throw std::invalid_argument("format is NULL");
        const auto doc = oxr_document::to_marker_document(ctx->document, "oxr-document");
        ctx->rendered = crisp_marker::render_document(doc, format);
        return ctx->rendered.c_str();
    } catch (const std::exception & error) {
        ctx->error = error.what();
        return nullptr;
    }
}

} // extern C
