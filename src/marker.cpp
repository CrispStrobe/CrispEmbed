// SPDX-License-Identifier: Apache-2.0
// Native Marker-style document conversion. See NOTICE.marker for provenance.
#include "crispembed.h"
#include "marker_backend.h"
#include "marker_document.h"
#include "core/env_gate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

#ifndef CRISPEMBED_VERSION_STR
#define CRISPEMBED_VERSION_STR "unknown"
#endif

struct crispembed_marker_context {
    crisp_marker::options opts;
    std::unique_ptr<crisp_marker::teleocr_backend> backend;
    crisp_marker::document doc;
    std::string error, output, metadata;
    bool converted = false;
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
}

template <typename F> int convert_guard(crispembed_marker_context * ctx, F fn) {
    if (!ctx) {
        init_error = "invalid Marker context";
        return 0;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
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
        if (c->opts.mode != 1) c->backend = std::make_unique<crisp_marker::teleocr_backend>(c->opts);
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
        const auto path = std::filesystem::u8path(file_path);
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open input file");
        char signature[1024]{};
        input.read(signature, sizeof(signature));
        const std::string prefix(signature, static_cast<size_t>(input.gcount()));
        c.doc.source_name = path.filename().u8string();
        if (prefix.find("%PDF-") != std::string::npos) {
            pdf_provider pdf(c.opts.pdfium_path, file_path, c.opts.password);
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
