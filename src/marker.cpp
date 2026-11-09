// SPDX-License-Identifier: Apache-2.0
// Native Marker-style document conversion. See NOTICE.marker for provenance.
#include "crispembed.h"
#include "marker_backend.h"
#include "marker_document.h"
#include "marker_artifact.h"
#include "core/env_gate.h"
#include "core/json.h"
#include "core/sha256.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

#ifndef CRISPEMBED_VERSION_STR
#define CRISPEMBED_VERSION_STR "unknown"
#endif
#ifndef CRISPEMBED_MARKER_BUILD_ID
#define CRISPEMBED_MARKER_BUILD_ID "unrecorded"
#endif
#ifndef CRISPEMBED_MARKER_BUILD_CONFIG
#define CRISPEMBED_MARKER_BUILD_CONFIG "unrecorded"
#endif

struct marker_job {
    crisp_marker::extraction_snapshot snapshot;
    std::unique_ptr<crisp_marker::pdf_provider> pdf;
    std::string input_path, checkpoint_path;
    crispembed_marker_job_status status = CRISPEMBED_MARKER_JOB_RUNNING;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
};

struct crispembed_marker_context {
    crisp_marker::options opts;
    std::unique_ptr<crisp_marker::teleocr_backend> backend;
    crisp_marker::document doc;
    std::string error, output, metadata;
    bool converted = false;
    std::unique_ptr<marker_job> job;
    std::atomic<bool> cancel_requested{ false };
    std::string progress, loaded_model_sha256, loaded_runtime_sha256, source_path;
    std::map<std::string, std::string> extraction_identity;
    int extraction_first = 0, extraction_last = -1, extraction_next = 0;
    bool extraction_complete = false;
    std::mutex mutex;
};

namespace {
thread_local std::string init_error;
using namespace crisp_marker;

void clear_result(crispembed_marker_context & ctx) {
    ctx.doc = document{};
    ctx.output.clear();
    ctx.metadata.clear();
    ctx.error.clear();
    ctx.converted = false;
    ctx.extraction_identity.clear();
    ctx.source_path.clear();
    ctx.extraction_first = ctx.extraction_next = 0;
    ctx.extraction_last = -1;
    ctx.extraction_complete = false;
}

void require_artifacts() {
    if (!core_env::on("CRISPEMBED_MARKER") || !core_env::on("CRISPEMBED_MARKER_ARTIFACTS"))
        throw std::runtime_error("extraction artifacts require CRISPEMBED_MARKER=1 and CRISPEMBED_MARKER_ARTIFACTS=1");
}

std::string processing_fingerprint(const options & o) {
    std::ostringstream s;
    s << "crispembed.marker-processing/1\n"
      << CRISPEMBED_VERSION_STR << '\n'
      << CRISPEMBED_MARKER_BUILD_ID << '\n'
      << CRISPEMBED_MARKER_BUILD_CONFIG << '\n'
      << o.mode << ',' << o.n_threads << ',' << o.first_page << ',' << o.last_page << ',' << o.dpi << ','
      << o.max_tokens << ',' << o.max_layout_tokens << ',' << o.max_blocks << ',' << o.max_pages << ',' << o.max_pixels
      << ',' << o.max_asset_bytes << ',' << o.include_page_headers << ',' << o.extract_images;
    // Values are parsed identically to their production paths; passwords and
    // absolute paths are deliberately absent from the serialized identity.
    s << ',' << core_env::on("CRISPEMBED_MARKER_REPETITION_CHECK") << ','
      << core_env::on("CRISPEMBED_MARKER_SOURCE_LINKS") << ','
      << core_env::on("CRISPEMBED_MARKER_MULTILINGUAL_NORMALIZE") << ','
      << core_env::on("CRISPEMBED_MARKER_REFERENCES");
    const auto text = s.str();
    return core_sha256::bytes(text.data(), text.size());
}

std::string runtime_fingerprint() {
    std::ostringstream s;
    // Hash the TeleOCR runtime settings, not paths or diagnostic output files.
    // Raw values conservatively distinguish unset from explicit overrides.
    for (const char * name :
         { "QWEN2VL_OCR_FORCE_CPU", "QWEN2VL_OCR_NO_WINDOW", "CRISPEMBED_TELEOCR_GPU", "CRISPEMBED_QWEN2VL_METAL_FA",
           "CRISPEMBED_QWEN2VL_METAL_SPLIT_K", "CRISPEMBED_TELEOCR_METAL_F32_MATMUL", "CRISPEMBED_NO_KV_CACHE",
           "CRISPEMBED_TELEOCR_REUSE_DECODE_GRAPH", "CRISPEMBED_QWEN2VL_NO_REPEAT_NGRAM",
           "CRISPEMBED_BPE_LEGACY_WHITESPACE", "CRISPEMBED_MAX_PIXELS" }) {
        const char * raw = std::getenv(name);
        const std::string value = raw ? raw : "";
        s << name << ':' << (raw ? "set:" : "unset:") << value.size() << ':' << value << '\n';
    }
    const auto text = s.str();
    return core_sha256::bytes(text.data(), text.size());
}

std::map<std::string, std::string> identity_for(crispembed_marker_context & c, const std::string & input) {
    std::map<std::string, std::string> identity;
    if (!input.empty()) identity["source_sha256"] = extraction_file_sha256(input);
    identity["model_sha256"] = "none";
    if (c.opts.mode != 1) {
        if (c.loaded_model_sha256.empty())
            throw std::runtime_error(
                "recreate the Marker context with the artifact gate enabled before loading a model");
        identity["model_sha256"] = extraction_file_sha256(c.opts.model_path);
        if (identity["model_sha256"] != c.loaded_model_sha256)
            throw std::runtime_error("model file changed since Marker initialization");
        if (runtime_fingerprint() != c.loaded_runtime_sha256)
            throw std::runtime_error("TeleOCR runtime settings changed; recreate the Marker context");
        for (const char * name :
             { "CRISPEMBED_QWEN2VL_FORCE_IDS", "CRISPEMBED_QWEN2VL_FORCE_FROM_CALL", "CRISPEMBED_QWEN2VL_REF" }) {
            const char * raw = std::getenv(name);
            const std::string value = raw ? raw : "";
            if (!value.empty())
                throw std::runtime_error(
                    "extraction artifacts do not admit stateful reference/forced-token diagnostics");
        }
    }
    identity["runtime_sha256"] = c.opts.mode == 1 ? "none" : c.loaded_runtime_sha256;
    identity["pdfium_sha256"] = c.opts.pdfium_path.empty() ? "unrecorded" : extraction_file_sha256(c.opts.pdfium_path);
    identity["processing_sha256"] = processing_fingerprint(c.opts);
    identity["build_sha256"] = CRISPEMBED_MARKER_BUILD_ID;
    identity["contract"] = "crispembed.marker-extraction/1";
    return identity;
}

bool same_path(const std::string & a, const std::string & b) {
    if (a.empty() || b.empty()) return false;
    const auto left = std::filesystem::absolute(std::filesystem::u8path(a)).lexically_normal();
    const auto right = std::filesystem::absolute(std::filesystem::u8path(b)).lexically_normal();
    if (left == right) return true;
    std::error_code error;
    return std::filesystem::equivalent(left, right, error) && !error;
}

void check_output_path(const crispembed_marker_context & c, const std::string & path, const std::string & input) {
    if (path.empty()) throw std::runtime_error("extraction artifact path is required");
    if (same_path(path, input) || same_path(path, c.opts.model_path) || same_path(path, c.opts.pdfium_path))
        throw std::runtime_error("extraction artifact must not replace the input, model or PDFium library");
}

artifact_limits context_artifact_limits(const crispembed_marker_context & c) {
    artifact_limits limits;
    limits.max_pages = std::min(limits.max_pages, static_cast<size_t>(c.opts.max_pages));
    limits.max_asset_bytes = std::min<uint64_t>(limits.max_asset_bytes, c.opts.max_asset_bytes);
    // max_blocks governs extracted top-level regions. Grouping/list splitting
    // adds descendants; the artifact's separate 4096-node structural cap applies.
    return limits;
}

options checked_options(const crispembed_marker_config * c) {
    if (!c || c->struct_size != sizeof(*c))
        throw std::runtime_error("invalid Marker config size; use default_config()");
    if (c->mode < 0 || c->mode > 2) throw std::runtime_error("Marker mode must be 0, 1 or 2");
    if (c->n_threads < 1 || c->n_threads > 64) throw std::runtime_error("Marker n_threads must be 1..64");
    if (c->first_page < 0 || c->last_page < -1 || (c->last_page >= 0 && c->last_page < c->first_page))
        throw std::runtime_error("invalid Marker page range");
    if (c->dpi < 36 || c->dpi > 600) throw std::runtime_error("Marker DPI must be 36..600");
    if (c->max_tokens < 1 || c->max_tokens > 8192 || c->max_layout_tokens < 1 || c->max_layout_tokens > 8192)
        throw std::runtime_error("Marker token limits must be 1..8192");
    if (c->max_blocks < 1 || c->max_blocks > 4096 || c->max_pages < 1 || c->max_pages > 10000)
        throw std::runtime_error("invalid Marker page/block limit");
    if (c->max_pixels < 1 || c->max_pixels > 64000000 || c->max_asset_bytes > 1024ULL * 1024 * 1024)
        throw std::runtime_error("invalid Marker pixel/asset limit");
    if ((c->include_page_headers != 0 && c->include_page_headers != 1) ||
        (c->extract_images != 0 && c->extract_images != 1))
        throw std::runtime_error("Marker boolean options must be 0 or 1");
    options o;
    o.model_path = c->model_path ? c->model_path : "";
    o.pdfium_path = c->pdfium_path ? c->pdfium_path : "";
    o.password = c->password ? c->password : "";
    o.mode = c->mode;
    o.n_threads = c->n_threads;
    o.first_page = c->first_page;
    o.last_page = c->last_page;
    o.dpi = c->dpi;
    o.max_tokens = c->max_tokens;
    o.max_layout_tokens = c->max_layout_tokens;
    o.max_blocks = c->max_blocks;
    o.max_pages = c->max_pages;
    o.max_pixels = static_cast<size_t>(c->max_pixels);
    o.max_asset_bytes = static_cast<size_t>(c->max_asset_bytes);
    o.include_page_headers = c->include_page_headers != 0;
    o.extract_images = c->extract_images != 0;
    if (o.mode != 1 && o.model_path.empty())
        throw std::runtime_error("a local TeleOCR GGUF is required in auto/force-OCR mode");
    return o;
}

std::vector<point> rect_polygon(const box & b) {
    return { { b.x0, b.y0 }, { b.x1, b.y0 }, { b.x1, b.y1 }, { b.x0, b.y1 } };
}

bool image_block(const block & b) {
    return b.type == "Picture" || b.type == "Figure" || b.type == "Diagram";
}

double overlap(const box & a, const box & b) {
    const double w = std::max(0.0, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
    const double h = std::max(0.0, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
    return w * h;
}

void assign_pdf_text(page & p) {
    // PDFium can return a visual row containing spans from multiple columns.
    // Assign each span once, retaining within-line order/font information.
    // Preserve unmatched text so missed layout regions do not erase content.
    std::vector<line> unmatched;
    for (const auto & l : p.text_lines) {
        std::map<int, line> assigned;
        for (const auto & s : l.spans) {
            if (s.text.empty()) continue;
            double best = 0, best_box_area = std::numeric_limits<double>::infinity();
            int selected = -1;
            const double area = std::max(0.001, s.bbox.width() * s.bbox.height());
            for (size_t i = 0; i < p.blocks.size(); ++i) {
                if (image_block(p.blocks[i])) continue;
                const auto & b = p.blocks[i].bbox;
                double score = overlap(s.bbox, b) / area;
                const double x = (s.bbox.x0 + s.bbox.x1) * 0.5, y = (s.bbox.y0 + s.bbox.y1) * 0.5;
                if (x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1) score = std::max(score, 0.51);
                const double box_area = b.width() * b.height();
                if (score > best || (score == best && score > 0 && box_area < best_box_area)) {
                    best = score;
                    best_box_area = box_area;
                    selected = static_cast<int>(i);
                }
            }
            if (best < 0.5) selected = -1;
            auto & part = assigned[selected];
            if (part.spans.empty())
                part.bbox = s.bbox;
            else
                part.bbox = { std::min(part.bbox.x0, s.bbox.x0), std::min(part.bbox.y0, s.bbox.y0),
                              std::max(part.bbox.x1, s.bbox.x1), std::max(part.bbox.y1, s.bbox.y1) };
            part.spans.push_back(s);
        }
        for (auto & item : assigned) {
            if (item.first >= 0)
                p.blocks[static_cast<size_t>(item.first)].lines.push_back(std::move(item.second));
            else
                unmatched.push_back(std::move(item.second));
        }
    }
    for (auto & b : p.blocks) {
        for (const auto & l : b.lines) {
            if (!b.text.empty()) b.text += '\n';
            b.text += text_of(l);
        }
        if (!b.text.empty()) b.source = "pdftext";
    }
    if (!unmatched.empty()) {
        page extra;
        extra.width = p.width;
        extra.height = p.height;
        extra.index = p.index;
        extra.text_lines = std::move(unmatched);
        build_text_blocks(extra);
        for (auto & b : extra.blocks) {
            b.order = static_cast<int>(p.blocks.size());
            b.warnings.push_back(
                "Embedded PDF text outside predicted layout retained; appended after predicted reading order.");
            p.blocks.push_back(std::move(b));
        }
    }
}

void extract_page(crispembed_marker_context & ctx, page & p, const image * raster) {
    if (ctx.opts.mode == 1) {
        build_text_blocks(p);
        p.extraction_method = "pdftext";
        if (p.text_lines.empty())
            p.warnings.push_back("No embedded PDF text; digital-text mode does not recognize scans.");
        if (p.blocks.size() > static_cast<size_t>(ctx.opts.max_blocks))
            throw std::runtime_error("page exceeds Marker block limit");
        return;
    }
    if (!raster) throw std::runtime_error("raster image required for TeleOCR");
    p.blocks = ctx.backend->layout(*raster, p.width, p.height, p.warnings, &p.raw_layout, &p.layout_generated_tokens,
                                   &p.layout_token_cap_reached);
    if (p.blocks.empty()) {
        if (ctx.opts.mode == 0 && !p.text_lines.empty()) {
            build_text_blocks(p);
            p.warnings.push_back("No valid TeleOCR layout regions; using embedded-text heuristic layout.");
        } else {
            block b;
            b.bbox = { 0, 0, p.width, p.height };
            b.polygon = rect_polygon(b.bbox);
            b.source = "fallback";
            b.warnings.push_back("No valid TeleOCR layout regions; recognizing the full page as text.");
            p.blocks.push_back(std::move(b));
        }
    } else if (ctx.opts.mode == 0) {
        assign_pdf_text(p);
    }
    if (p.blocks.size() > static_cast<size_t>(ctx.opts.max_blocks))
        throw std::runtime_error("page exceeds Marker block limit");
    bool used_ocr = false, used_pdf = false;
    for (size_t i = 0; i < p.blocks.size(); ++i) {
        auto & b = p.blocks[i];
        if (image_block(b)) {
            if (ctx.opts.extract_images) {
                b.image_name = "page_" + std::to_string(p.index) + "_figure_" + std::to_string(i) + ".png";
                asset a = encode_image_asset(crop_image(*raster, b, p.width, p.height), b.image_name);
                size_t total = a.bytes.size();
                for (const auto & existing : ctx.doc.assets) {
                    if (existing.bytes.size() > ctx.opts.max_asset_bytes - std::min(total, ctx.opts.max_asset_bytes))
                        throw std::runtime_error("Marker image assets exceed byte limit");
                    total += existing.bytes.size();
                }
                if (total > ctx.opts.max_asset_bytes) throw std::runtime_error("Marker image assets exceed byte limit");
                ctx.doc.assets.push_back(std::move(a));
            }
            continue;
        }
        bool use_pdf = ctx.opts.mode == 0 && text_is_usable(b.text);
        if (use_pdf && b.type == "Table") {
            b.html = pdf_table_html(b.lines, b.warnings);
            use_pdf = !b.html.empty();
        }
        // Embedded text does not encode mathematical structure reliably.
        if (b.type == "Equation" || b.type == "ChemicalBlock") use_pdf = false;
        if (use_pdf) {
            b.source = "pdftext";
            used_pdf = true;
        } else {
            ctx.backend->recognize(*raster, b, p.width, p.height);
            used_ocr = true;
        }
    }
    p.extraction_method = used_ocr ? (used_pdf ? "pdftext+teleocr" : "teleocr") : "pdftext";
}

void finish(crispembed_marker_context & ctx, std::chrono::steady_clock::time_point start) {
    process_document(ctx.doc, ctx.opts);
    ctx.doc.metadata["implementation"] = "crispembed-marker-native";
    ctx.doc.metadata["software_version"] = CRISPEMBED_VERSION_STR;
    ctx.doc.metadata["marker_source_revision"] = "71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4";
    ctx.doc.metadata["schema"] = "crispembed.marker/1";
    ctx.doc.metadata["surya_model_parity"] = "not_claimed";
    ctx.doc.metadata["text_quality_check"] = "deterministic_unicode_heuristic";
    ctx.doc.metadata["layout_backend"] = ctx.opts.mode == 1 ? "pdftext_heuristics" : "teleocr_ggml";
    if (!ctx.opts.model_path.empty() && ctx.opts.mode != 1)
        ctx.doc.metadata["model_file"] = std::filesystem::path(ctx.opts.model_path).filename().string();
    ctx.doc.metadata["elapsed_seconds"] =
        std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    ctx.metadata = render_metadata(ctx.doc);
    ctx.converted = true;
    ctx.extraction_first = ctx.doc.pages.empty() ? ctx.opts.first_page : ctx.doc.pages.front().index;
    ctx.extraction_last = ctx.doc.pages.empty() ? ctx.extraction_first - 1 : ctx.doc.pages.back().index;
    ctx.extraction_next = ctx.extraction_last + 1;
    ctx.extraction_complete = true;
}

template <typename F> int convert_guard(crispembed_marker_context * ctx, F fn) {
    if (!ctx) {
        init_error = "invalid Marker context";
        return 0;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->job.reset();
    ctx->cancel_requested.store(false);
    ctx->progress.clear();
    clear_result(*ctx);
    const auto start = std::chrono::steady_clock::now();
    try {
        if (!core_env::on("CRISPEMBED_MARKER")) throw std::runtime_error("Marker requires CRISPEMBED_MARKER=1");
        if (ctx->opts.mode != 1 && !core_env::on("CRISPEMBED_TELEOCR"))
            throw std::runtime_error("TeleOCR requires CRISPEMBED_TELEOCR=1");
        // TeleOCR's vision tower is under the Qwen RESEARCH LICENSE (docs/teleocr-license.md);
        // the PDFium text mode loads no model and needs no acknowledgement.
        if (ctx->opts.mode != 1 && !core_env::on("CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE"))
            throw std::runtime_error("TeleOCR's vision tower is under the Qwen RESEARCH LICENSE (non-commercial); "
                                     "set CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1 to acknowledge");
        ctx->doc.backend = ctx->opts.mode == 1 ? "pdfium-text" : "crispembed-teleocr";
        fn(*ctx);
        finish(*ctx, start);
        return 1;
    } catch (const std::exception & e) {
        clear_result(*ctx);
        ctx->error = e.what();
    } catch (...) {
        clear_result(*ctx);
        ctx->error = "unexpected Marker conversion failure";
    }
    return 0;
}
} // namespace

extern "C" crispembed_marker_config crispembed_marker_default_config(void) {
    crispembed_marker_config c{};
    c.struct_size = sizeof(c);
    c.n_threads = 4;
    c.last_page = -1;
    c.dpi = 144;
    c.max_tokens = 1024;
    c.max_layout_tokens = 2048;
    c.max_blocks = 512;
    c.max_pages = 1000;
    c.max_pixels = 16000000;
    c.max_asset_bytes = 128 * 1024 * 1024;
    c.extract_images = 1;
    return c;
}
extern "C" crispembed_marker_context * crispembed_marker_init(const crispembed_marker_config * config) {
    init_error.clear();
    try {
        if (!core_env::on("CRISPEMBED_MARKER")) throw std::runtime_error("Marker requires CRISPEMBED_MARKER=1");
        auto c = std::make_unique<crispembed_marker_context>();
        c->opts = checked_options(config);
        if (c->opts.mode != 1) {
            if (core_env::on("CRISPEMBED_MARKER_ARTIFACTS")) {
                if (!core_env::on("CRISPEMBED_TELEOCR") || !core_env::on("CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE"))
                    throw std::runtime_error(
                        "TeleOCR requires feature enablement and Qwen Research acknowledgement before model access");
                c->loaded_model_sha256 = extraction_file_sha256(c->opts.model_path);
                c->loaded_runtime_sha256 = runtime_fingerprint();
            }
            c->backend = std::make_unique<crisp_marker::teleocr_backend>(c->opts);
            if (!c->loaded_model_sha256.empty() && extraction_file_sha256(c->opts.model_path) != c->loaded_model_sha256)
                throw std::runtime_error("model file changed while loading Marker");
        }
        return c.release();
    } catch (const std::exception & e) {
        init_error = e.what();
    } catch (...) {
        init_error = "unexpected Marker initialization failure";
    }
    return nullptr;
}
extern "C" void crispembed_marker_free(crispembed_marker_context * ctx) {
    delete ctx;
}

extern "C" int crispembed_marker_convert(crispembed_marker_context * ctx, const char * file_path) {
    return convert_guard(ctx, [&](crispembed_marker_context & c) {
        if (!file_path || !*file_path) throw std::runtime_error("input file path is required");
        c.source_path = file_path;
        if (core_env::on("CRISPEMBED_MARKER_ARTIFACTS")) c.extraction_identity = identity_for(c, file_path);
        const auto path = std::filesystem::u8path(file_path);
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open input file");
        char signature[1024]{};
        input.read(signature, sizeof(signature));
        const std::string prefix(signature, static_cast<size_t>(input.gcount()));
        c.doc.source_name = path.filename().u8string();
        if (prefix.find("%PDF-") != std::string::npos) {
            pdf_provider pdf(c.opts.pdfium_path, file_path, c.opts.password);
            if (!c.extraction_identity.empty() && pdf.library_sha256() != c.extraction_identity.at("pdfium_sha256"))
                throw std::runtime_error("loaded PDFium identity differs from the extraction identity");
            const int count = pdf.page_count();
            const int last = c.opts.last_page < 0 ? count - 1 : c.opts.last_page;
            if (c.opts.first_page >= count || last >= count)
                throw std::runtime_error("PDF page range exceeds document");
            if (last - c.opts.first_page + 1 > c.opts.max_pages)
                throw std::runtime_error("PDF exceeds selected page limit");
            for (int i = c.opts.first_page; i <= last; ++i) {
                page p = pdf.read_page(i, c.opts.mode != 2);
                image raster;
                if (c.opts.mode != 1) raster = pdf.render_page(i, c.opts.dpi, c.opts.max_pixels);
                extract_page(c, p, c.opts.mode == 1 ? nullptr : &raster);
                c.doc.pages.push_back(std::move(p));
            }
        } else {
            if (c.opts.mode == 1)
                throw std::runtime_error("digital-text mode requires a PDF text layer; use auto/force-OCR for images");
            if (c.opts.first_page != 0 || c.opts.last_page > 0)
                throw std::runtime_error("image input contains only page 0");
            image raster = load_image(file_path, c.opts.max_pixels);
            page p;
            p.width = raster.width;
            p.height = raster.height;
            extract_page(c, p, &raster);
            c.doc.pages.push_back(std::move(p));
        }
    });
}

extern "C" int crispembed_marker_convert_raw(crispembed_marker_context * ctx, const uint8_t * pixels, int width,
                                             int height, int channels) {
    return convert_guard(ctx, [&](crispembed_marker_context & c) {
        if (c.opts.mode == 1) throw std::runtime_error("digital-text mode does not recognize raw images");
        if (!pixels || width < 1 || height < 1 || width > 32768 || height > 32768 ||
            (channels != 1 && channels != 3 && channels != 4))
            throw std::runtime_error("invalid raw image dimensions/channels");
        if (static_cast<uint64_t>(width) * static_cast<uint64_t>(height) > c.opts.max_pixels)
            throw std::runtime_error("raw image exceeds pixel limit");
        if (c.opts.first_page != 0 || c.opts.last_page > 0) throw std::runtime_error("raw input contains only page 0");
        if (core_env::on("CRISPEMBED_MARKER_ARTIFACTS")) {
            c.extraction_identity = identity_for(c, "");
            core_sha256::sha256_state hash;
            const auto dimensions =
                std::to_string(width) + "," + std::to_string(height) + "," + std::to_string(channels) + ":";
            core_sha256::sha256_update(hash, reinterpret_cast<const uint8_t *>(dimensions.data()), dimensions.size());
            core_sha256::sha256_update(
                hash, pixels, static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels));
            c.extraction_identity["source_sha256"] = core_sha256::sha256_finish(hash);
            c.extraction_identity["source_kind"] = "raw-image";
        }
        image raster;
        raster.width = width;
        raster.height = height;
        const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
        raster.rgb.resize(n * 3);
        for (size_t i = 0; i < n; ++i) {
            for (int k = 0; k < 3; ++k) {
                const uint8_t v = pixels[i * static_cast<size_t>(channels) + (channels == 1 ? 0 : k)];
                const unsigned alpha = channels == 4 ? pixels[i * 4 + 3] : 255;
                raster.rgb[i * 3 + k] = static_cast<uint8_t>((v * alpha + 255 * (255 - alpha) + 127) / 255);
            }
        }
        c.doc.source_name = "raw-image";
        page p;
        p.width = width;
        p.height = height;
        extract_page(c, p, &raster);
        c.doc.pages.push_back(std::move(p));
    });
}

namespace {
void require_incremental(const crispembed_marker_context & c) {
    require_artifacts();
    if (!core_env::on("CRISPEMBED_MARKER_INCREMENTAL"))
        throw std::runtime_error("incremental extraction requires CRISPEMBED_MARKER_INCREMENTAL=1");
    if (c.opts.mode != 1 &&
        (!core_env::on("CRISPEMBED_TELEOCR") || !core_env::on("CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE")))
        throw std::runtime_error("incremental TeleOCR requires feature enablement and Qwen Research acknowledgement");
}

void job_view(crispembed_marker_context & c) {
    c.doc = c.job->snapshot.value;
    process_document(c.doc, c.opts);
    c.doc.metadata["implementation"] = "crispembed-marker-native";
    c.doc.metadata["software_version"] = CRISPEMBED_VERSION_STR;
    c.doc.metadata["extraction_state"] = c.job->snapshot.complete ? "complete" : "partial";
    c.doc.metadata["completed_pages"] = std::to_string(c.job->snapshot.value.pages.size());
    c.doc.metadata["next_page"] = std::to_string(c.job->snapshot.next_page);
    c.doc.metadata["extraction_phase"] = "processed";
    c.extraction_identity = c.job->snapshot.identity;
    c.source_path = c.job->input_path;
    c.extraction_first = c.job->snapshot.first_page;
    c.extraction_last = c.job->snapshot.last_page;
    c.extraction_next = c.job->snapshot.next_page;
    c.extraction_complete = c.job->snapshot.complete;
    c.output.clear();
    c.metadata = render_metadata(c.doc);
    c.converted = true;
}

void checkpoint(marker_job & job, const artifact_limits & limits) {
    job.snapshot.value.metadata["extraction_phase"] = "raw";
    job.snapshot.value.metadata["extraction_state"] = job.snapshot.complete ? "complete" : "partial";
    save_extraction(job.checkpoint_path, job.snapshot, limits);
}
} // namespace

extern "C" int crispembed_marker_start(crispembed_marker_context * ctx, const char * file_path,
                                       const char * checkpoint_path, int resume) {
    if (!ctx) {
        init_error = "invalid Marker context";
        return 0;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->error.clear();
    try {
        require_incremental(*ctx);
        if (!file_path || !*file_path || !checkpoint_path || !*checkpoint_path || (resume != 0 && resume != 1))
            throw std::runtime_error("input/checkpoint paths and resume=0/1 are required");
        check_output_path(*ctx, checkpoint_path, file_path);
        if (std::string(CRISPEMBED_MARKER_BUILD_ID) == "unrecorded")
            throw std::runtime_error("incremental extraction requires a source-identified native build");
        auto next = std::make_unique<marker_job>();
        next->input_path = file_path;
        next->checkpoint_path = checkpoint_path;
        const auto identity = identity_for(*ctx, next->input_path);
        std::ifstream input(std::filesystem::u8path(file_path), std::ios::binary);
        char signature[1024]{};
        input.read(signature, sizeof(signature));
        const bool is_pdf =
            std::string(signature, static_cast<size_t>(input.gcount())).find("%PDF-") != std::string::npos;
        int last = 0;
        if (is_pdf) {
            if (ctx->opts.pdfium_path.empty())
                throw std::runtime_error(
                    "incremental PDF extraction requires an explicit PDFium library for identity validation");
            next->pdf = std::make_unique<pdf_provider>(ctx->opts.pdfium_path, file_path, ctx->opts.password);
            if (next->pdf->library_sha256() != identity.at("pdfium_sha256"))
                throw std::runtime_error("loaded PDFium identity differs from the checkpoint identity");
            const int count = next->pdf->page_count();
            last = ctx->opts.last_page < 0 ? count - 1 : ctx->opts.last_page;
            if (ctx->opts.first_page >= count || last >= count)
                throw std::runtime_error("PDF page range exceeds document");
        } else {
            if (ctx->opts.mode == 1) throw std::runtime_error("digital-text mode requires a PDF text layer");
            if (ctx->opts.first_page != 0 || ctx->opts.last_page > 0)
                throw std::runtime_error("image input contains only page 0");
        }
        if (last - ctx->opts.first_page + 1 > std::min(ctx->opts.max_pages, 1000))
            throw std::runtime_error("incremental extraction exceeds selected page or artifact limit");
        if (resume) {
            next->snapshot = load_extraction(checkpoint_path, context_artifact_limits(*ctx));
            if (next->snapshot.identity != identity || next->snapshot.first_page != ctx->opts.first_page ||
                next->snapshot.last_page != last)
                throw std::runtime_error("checkpoint input, model, processing identity or selection does not match");
            if (next->snapshot.value.metadata["extraction_phase"] != "raw")
                throw std::runtime_error("resume requires a raw extraction checkpoint, not a rendered-view artifact");
        } else {
            if (std::filesystem::exists(std::filesystem::u8path(checkpoint_path)))
                throw std::runtime_error("checkpoint already exists; choose resume=1 or a new checkpoint path");
            next->snapshot.identity = identity;
            next->snapshot.first_page = ctx->opts.first_page;
            next->snapshot.last_page = last;
            next->snapshot.next_page = ctx->opts.first_page;
            next->snapshot.value.source_name = std::filesystem::u8path(file_path).filename().u8string();
            next->snapshot.value.backend = ctx->opts.mode == 1 ? "pdfium-text" : "crispembed-teleocr";
            checkpoint(*next, context_artifact_limits(*ctx));
        }
        next->status = next->snapshot.complete ? CRISPEMBED_MARKER_JOB_COMPLETE : CRISPEMBED_MARKER_JOB_RUNNING;
        clear_result(*ctx);
        ctx->job = std::move(next);
        ctx->cancel_requested.store(false);
        job_view(*ctx);
        return 1;
    } catch (const std::exception & e) {
        ctx->error = e.what();
    } catch (...) {
        ctx->error = "unexpected incremental Marker initialization failure";
    }
    return 0;
}

extern "C" crispembed_marker_job_status crispembed_marker_step(crispembed_marker_context * ctx, int page_budget) {
    if (!ctx) {
        init_error = "invalid Marker context";
        return CRISPEMBED_MARKER_JOB_ERROR;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->error.clear();
    try {
        require_incremental(*ctx);
        if (!ctx->job) throw std::runtime_error("no incremental Marker job; call start first");
        if (page_budget < 1 || page_budget > 64) throw std::runtime_error("Marker page budget must be 1..64");
        auto & job = *ctx->job;
        if (job.status == CRISPEMBED_MARKER_JOB_COMPLETE || job.status == CRISPEMBED_MARKER_JOB_CANCELLED)
            return job.status;
        if (job.status == CRISPEMBED_MARKER_JOB_ERROR)
            throw std::runtime_error("failed job must be resumed from its checkpoint");
        if (identity_for(*ctx, job.input_path) != job.snapshot.identity)
            throw std::runtime_error("input, model, PDFium or processing identity changed during extraction");
        for (int n = 0; n < page_budget && !job.snapshot.complete; ++n) {
            if (ctx->cancel_requested.load()) {
                job.status = CRISPEMBED_MARKER_JOB_CANCELLED;
                break;
            }
            page p;
            image raster;
            const int index = job.snapshot.next_page;
            if (job.pdf) {
                p = job.pdf->read_page(index, ctx->opts.mode != 2);
                if (ctx->opts.mode != 1) raster = job.pdf->render_page(index, ctx->opts.dpi, ctx->opts.max_pixels);
            } else {
                raster = load_image(job.input_path, ctx->opts.max_pixels);
                p.index = index;
                p.width = raster.width;
                p.height = raster.height;
            }
            // Extract into the durable raw document. On page failure, discard
            // only assets created by that page. Previously committed units stay.
            ctx->doc = std::move(job.snapshot.value);
            const size_t asset_count = ctx->doc.assets.size();
            try {
                extract_page(*ctx, p, ctx->opts.mode == 1 ? nullptr : &raster);
                ctx->doc.pages.push_back(std::move(p));
            } catch (...) {
                ctx->doc.assets.resize(asset_count);
                job.snapshot.value = std::move(ctx->doc);
                throw;
            }
            job.snapshot.value = std::move(ctx->doc);
            ++job.snapshot.next_page;
            job.snapshot.complete = job.snapshot.next_page > job.snapshot.last_page;
            try {
                checkpoint(job, context_artifact_limits(*ctx));
            } catch (...) {
                job.snapshot.value.pages.pop_back();
                job.snapshot.value.assets.resize(asset_count);
                --job.snapshot.next_page;
                job.snapshot.complete = false;
                throw;
            }
            if (ctx->cancel_requested.load() && !job.snapshot.complete) job.status = CRISPEMBED_MARKER_JOB_CANCELLED;
            if (job.status == CRISPEMBED_MARKER_JOB_CANCELLED) break;
        }
        if (job.snapshot.complete) job.status = CRISPEMBED_MARKER_JOB_COMPLETE;
        job_view(*ctx);
        return job.status;
    } catch (const std::exception & e) {
        ctx->error = e.what();
    } catch (...) {
        ctx->error = "unexpected incremental Marker extraction failure";
    }
    if (ctx->job) {
        ctx->job->status = CRISPEMBED_MARKER_JOB_ERROR;
        // Regenerate the view from completed units, retaining the primary error.
        try {
            job_view(*ctx);
        } catch (...) {
            ctx->converted = false;
        }
    }
    return CRISPEMBED_MARKER_JOB_ERROR;
}

extern "C" void crispembed_marker_cancel(crispembed_marker_context * ctx) {
    if (ctx) ctx->cancel_requested.store(true);
}

extern "C" const char * crispembed_marker_progress(crispembed_marker_context * ctx) {
    if (!ctx) {
        init_error = "invalid Marker context";
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    try {
        const auto * job = ctx->job.get();
        const char * state = !job                                             ? "idle"
                             : job->status == CRISPEMBED_MARKER_JOB_COMPLETE  ? "complete"
                             : job->status == CRISPEMBED_MARKER_JOB_CANCELLED ? "cancelled"
                             : job->status == CRISPEMBED_MARKER_JOB_ERROR     ? "error"
                                                                              : "running";
        ctx->progress =
            "{\"schema\":\"crispembed.marker-progress/1\",\"state\":\"" + std::string(state) +
            "\",\"completed_pages\":" + std::to_string(job ? job->snapshot.value.pages.size() : 0) +
            ",\"total_pages\":" + std::to_string(job ? job->snapshot.last_page - job->snapshot.first_page + 1 : 0) +
            ",\"next_page\":" + std::to_string(job ? job->snapshot.next_page : 0) +
            ",\"cancel_requested\":" + (ctx->cancel_requested.load() ? "true" : "false") + "}";
        return ctx->progress.c_str();
    } catch (const std::exception & e) {
        ctx->error = e.what();
    } catch (...) {
        ctx->error = "unexpected Marker progress failure";
    }
    return nullptr;
}

extern "C" int crispembed_marker_save_extraction(crispembed_marker_context * ctx, const char * path) {
    if (!ctx) {
        init_error = "invalid Marker context";
        return 0;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->error.clear();
    try {
        require_artifacts();
        if (!ctx->converted || !path || !*path)
            throw std::runtime_error("a converted view and extraction path are required");
        check_output_path(*ctx, path, ctx->source_path);
        if (ctx->extraction_identity.empty())
            throw std::runtime_error(
                "convert with the artifact gate enabled to record extraction identity before saving");
        if (ctx->job && same_path(path, ctx->job->checkpoint_path))
            throw std::runtime_error("save the rendered view separately from the resumable raw checkpoint");
        extraction_snapshot snapshot;
        snapshot.value = ctx->doc;
        snapshot.value.metadata["extraction_phase"] = "processed";
        snapshot.identity = ctx->extraction_identity;
        snapshot.first_page = ctx->extraction_first;
        snapshot.next_page = ctx->extraction_next;
        snapshot.last_page = ctx->extraction_last;
        snapshot.complete = ctx->extraction_complete;
        save_extraction(path, snapshot, context_artifact_limits(*ctx));
        return 1;
    } catch (const std::exception & e) {
        ctx->error = e.what();
    } catch (...) {
        ctx->error = "unexpected extraction save failure";
    }
    return 0;
}

extern "C" int crispembed_marker_open_extraction(crispembed_marker_context * ctx, const char * path) {
    if (!ctx) {
        init_error = "invalid Marker context";
        return 0;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->error.clear();
    try {
        require_artifacts();
        if (!path || !*path) throw std::runtime_error("extraction path is required");
        auto snapshot = load_extraction(path, context_artifact_limits(*ctx));
        if (snapshot.value.pages.size() > static_cast<size_t>(ctx->opts.max_pages))
            throw std::runtime_error("extraction exceeds configured page limit");
        size_t assets = 0;
        for (const auto & a : snapshot.value.assets) {
            if (a.bytes.size() > ctx->opts.max_asset_bytes - assets)
                throw std::runtime_error("extraction exceeds configured asset limit");
            assets += a.bytes.size();
        }
        const auto phase = snapshot.value.metadata.find("extraction_phase");
        if (phase == snapshot.value.metadata.end() || (phase->second != "raw" && phase->second != "processed"))
            throw std::runtime_error("extraction phase is missing or unsupported");
        if (phase->second == "raw") {
            if (snapshot.identity["processing_sha256"] != processing_fingerprint(ctx->opts))
                throw std::runtime_error("raw checkpoint requires matching processing settings; use a processed "
                                         "artifact for standalone viewing");
            process_document(snapshot.value, ctx->opts);
        }
        snapshot.value.metadata["extraction_state"] = snapshot.complete ? "complete" : "partial";
        snapshot.value.metadata["extraction_phase"] = "processed";
        const auto metadata = render_metadata(snapshot.value);
        ctx->job.reset();
        ctx->cancel_requested.store(false);
        ctx->source_path.clear();
        ctx->extraction_first = snapshot.first_page;
        ctx->extraction_last = snapshot.last_page;
        ctx->extraction_next = snapshot.next_page;
        ctx->extraction_complete = snapshot.complete;
        ctx->doc = std::move(snapshot.value);
        ctx->extraction_identity = std::move(snapshot.identity);
        ctx->metadata = metadata;
        ctx->output.clear();
        ctx->converted = true;
        return 1;
    } catch (const std::exception & e) {
        ctx->error = e.what();
    } catch (...) {
        ctx->error = "unexpected extraction open failure";
    }
    return 0;
}

extern "C" const char * crispembed_marker_render(crispembed_marker_context * ctx, const char * format) {
    if (!ctx) {
        init_error = "invalid Marker context";
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->output.clear();
    ctx->error.clear();
    try {
        if (!ctx->converted) throw std::runtime_error("no successful Marker conversion");
        if (!format) throw std::runtime_error("render format is required");
        ctx->output = crisp_marker::render_document(ctx->doc, format);
        return ctx->output.c_str();
    } catch (const std::exception & e) {
        ctx->error = e.what();
    } catch (...) {
        ctx->error = "unexpected Marker rendering failure";
    }
    return nullptr;
}
extern "C" const char * crispembed_marker_metadata(crispembed_marker_context * ctx) {
    if (!ctx) {
        init_error = "invalid Marker context";
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (!ctx->converted) {
        ctx->error = "no successful Marker conversion";
        return nullptr;
    }
    return ctx->metadata.c_str();
}
extern "C" const char * crispembed_marker_error(const crispembed_marker_context * ctx) {
    return ctx ? ctx->error.c_str() : init_error.c_str();
}
extern "C" int crispembed_marker_asset_count(const crispembed_marker_context * ctx) {
    return ctx && ctx->converted ? static_cast<int>(ctx->doc.assets.size()) : 0;
}
extern "C" const char * crispembed_marker_asset_name(const crispembed_marker_context * ctx, int index) {
    if (!ctx || !ctx->converted || index < 0 || static_cast<size_t>(index) >= ctx->doc.assets.size()) return nullptr;
    return ctx->doc.assets[static_cast<size_t>(index)].name.c_str();
}
extern "C" const uint8_t * crispembed_marker_asset_data(const crispembed_marker_context * ctx, int index,
                                                        size_t * out_size) {
    if (out_size) *out_size = 0;
    if (!ctx || !ctx->converted || index < 0 || static_cast<size_t>(index) >= ctx->doc.assets.size()) return nullptr;
    const auto & bytes = ctx->doc.assets[static_cast<size_t>(index)].bytes;
    if (out_size) *out_size = bytes.size();
    return bytes.data();
}
